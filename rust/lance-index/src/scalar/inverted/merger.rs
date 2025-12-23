// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use futures::future::try_join_all;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use snafu::location;
use std::sync::Arc;

use crate::scalar::{IndexStore, IndexWriter};

use super::{
    builder::{
        doc_file_path, inverted_list_schema, posting_file_path, token_file_path, InnerBuilder,
        PositionRecorder,
    },
    DocSet, InvertedPartition, PostingListBuilder, PostingListReader, TokenSet, TokenSetFormat,
};

enum MergeEvent {
    PartitionStart {
        tokens: TokenSet,
        docs: DocSet,
        inverted_list: Arc<PostingListReader>,
    },
    PostingBatch {
        token_id: u32,
        batch: RecordBatch,
    },
    PartitionEnd,
}

enum WriteOp {
    StartPartition { id: u64, with_position: bool },
    PostingBatch { id: u64, batch: RecordBatch },
    FinishPartition {
        id: u64,
        tokens_batch: RecordBatch,
        docs_batch: RecordBatch,
    },
}

struct PartitionContext {
    token_map: Vec<u32>,
    doc_id_offset: u32,
    inverted_list: Arc<PostingListReader>,
}

async fn write_partition_ops(
    store: &dyn IndexStore,
    receiver: async_channel::Receiver<WriteOp>,
) -> Result<()> {
    let mut current_id: Option<u64> = None;
    let mut posting_writer: Option<Box<dyn IndexWriter>> = None;

    while let Ok(op) = receiver.recv().await {
        match op {
            WriteOp::StartPartition { id, with_position } => {
                if current_id.is_some() {
                    return Err(Error::Internal {
                        message: "start received with active partition".to_owned(),
                        location: location!(),
                    });
                }
                let writer = store
                    .new_index_file(&posting_file_path(id), inverted_list_schema(with_position))
                    .await?;
                current_id = Some(id);
                posting_writer = Some(writer);
            }
            WriteOp::PostingBatch { id, batch } => {
                if current_id != Some(id) {
                    return Err(Error::Internal {
                        message: "posting batch received for inactive partition".to_owned(),
                        location: location!(),
                    });
                }
                let writer = posting_writer.as_mut().ok_or_else(|| Error::Internal {
                    message: "posting writer not initialized".to_owned(),
                    location: location!(),
                })?;
                writer.write_record_batch(batch).await?;
            }
            WriteOp::FinishPartition {
                id,
                tokens_batch,
                docs_batch,
            } => {
                if current_id != Some(id) {
                    return Err(Error::Internal {
                        message: "finish received for inactive partition".to_owned(),
                        location: location!(),
                    });
                }
                let mut writer = posting_writer.take().ok_or_else(|| Error::Internal {
                    message: "posting writer not initialized".to_owned(),
                    location: location!(),
                })?;
                writer.finish().await?;
                current_id = None;

                let mut token_writer = store
                    .new_index_file(&token_file_path(id), tokens_batch.schema())
                    .await?;
                token_writer.write_record_batch(tokens_batch).await?;
                token_writer.finish().await?;

                let mut doc_writer = store
                    .new_index_file(&doc_file_path(id), docs_batch.schema())
                    .await?;
                doc_writer.write_record_batch(docs_batch).await?;
                doc_writer.finish().await?;
            }
        }
    }

    Ok(())
}

pub trait Merger {
    // Merge the partitions and write new partitions,
    // the new partitions are returned.
    // This method would read all the input partitions at the same time,
    // so it's not recommended to pass too many partitions.
    async fn merge(&mut self) -> Result<Vec<u64>>;
}

// A merger that merges partitions based on their size,
// it would read the posting lists for each token from
// the partitions and write them to a new partition,
// until the size of the new partition reaches the target size.
pub struct SizeBasedMerger<'a> {
    dest_store: &'a dyn IndexStore,
    input: Vec<InvertedPartition>,
    with_position: bool,
    target_size: u64,
    token_set_format: TokenSetFormat,
    builder: InnerBuilder,
    partitions: Vec<u64>,
}

impl<'a> SizeBasedMerger<'a> {
    // Create a new SizeBasedMerger with the target size,
    // the size is compressed size in byte.
    // Typically, just set the size to the memory limit,
    // because less partitions means faster query.
    pub fn new(
        dest_store: &'a dyn IndexStore,
        input: Vec<InvertedPartition>,
        target_size: u64,
        token_set_format: TokenSetFormat,
    ) -> Self {
        let max_id = input.iter().map(|p| p.id()).max().unwrap_or(0);
        let with_position = input
            .first()
            .map(|p| p.inverted_list.has_positions())
            .unwrap_or(false);

        Self {
            dest_store,
            input,
            with_position,
            target_size,
            token_set_format,
            builder: InnerBuilder::new(max_id + 1, with_position, token_set_format),
            partitions: Vec::new(),
        }
    }
}

impl Merger for SizeBasedMerger<'_> {
    async fn merge(&mut self) -> Result<Vec<u64>> {
        if self.input.len() <= 1 {
            for part in self.input.iter() {
                part.store()
                    .copy_index_file(&token_file_path(part.id()), self.dest_store)
                    .await?;
                part.store()
                    .copy_index_file(&posting_file_path(part.id()), self.dest_store)
                    .await?;
                part.store()
                    .copy_index_file(&doc_file_path(part.id()), self.dest_store)
                    .await?;
            }

            return Ok(self.input.iter().map(|p| p.id()).collect());
        }

        // for token set, union the tokens,
        // for doc set, concatenate the row ids, assign the doc id to offset + doc_id
        // for posting list, concatenate the posting lists
        log::info!(
            "merging {} partitions with target size {} MiB",
            self.input.len(),
            self.target_size / 1024 / 1024
        );

        let num_workers = *crate::scalar::inverted::builder::LANCE_FTS_NUM_SHARDS;
        let (read_sender, read_receiver) = async_channel::bounded(2);

        let mut write_senders = Vec::with_capacity(num_workers);
        let mut writer_futures = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let (sender, receiver) = async_channel::bounded(2);
            write_senders.push(sender);
            let store = self.dest_store;
            writer_futures.push(async move { write_partition_ops(store, receiver).await });
        }

        let num_parts = self.input.len();
        let parts = std::mem::take(&mut self.input);
        let reader = async move {
            for part in parts {
                let InvertedPartition {
                    tokens,
                    inverted_list,
                    docs,
                    ..
                } = part;
                let num_tokens = tokens.len() as u32;
                let has_positions = inverted_list.has_positions();
                read_sender
                    .send(MergeEvent::PartitionStart {
                        tokens,
                        docs,
                        inverted_list: inverted_list.clone(),
                    })
                    .await
                    .map_err(|_| Error::Internal {
                        message: "merge input channel closed".to_owned(),
                        location: location!(),
                    })?;

                for token_id in 0..num_tokens {
                    if inverted_list.posting_len(token_id) == 0 {
                        continue;
                    }
                    let batch = inverted_list
                        .read_posting_batch(token_id, has_positions)
                        .await?;
                    read_sender
                        .send(MergeEvent::PostingBatch { token_id, batch })
                        .await
                        .map_err(|_| Error::Internal {
                            message: "merge input channel closed".to_owned(),
                            location: location!(),
                        })?;
                }

                read_sender
                    .send(MergeEvent::PartitionEnd)
                    .await
                    .map_err(|_| Error::Internal {
                        message: "merge input channel closed".to_owned(),
                        location: location!(),
                    })?;
            }
            Ok::<(), Error>(())
        };

        let with_position = self.with_position;
        let token_set_format = self.token_set_format;
        let target_size = self.target_size;
        let start = std::time::Instant::now();
        let builder_id = self.builder.id();
        let cpu = spawn_cpu(move || {
            let mut partitions = Vec::new();
            let mut builder = InnerBuilder::new(builder_id, with_position, token_set_format);
            let mut estimated_size = builder.docs.size() + builder.tokens.estimated_size();
            let mut parts_merged = 0usize;
            let mut current: Option<PartitionContext> = None;

            let send_write_op = |sender: &async_channel::Sender<WriteOp>,
                                 op: WriteOp|
             -> Result<()> {
                sender.send_blocking(op).map_err(|_| Error::Internal {
                    message: "partition channel closed".to_owned(),
                    location: location!(),
                })
            };

            let flush_builder = |builder: &mut InnerBuilder| -> Result<u64> {
                let id = builder.id();
                let docs = Arc::new(std::mem::take(&mut builder.docs));
                let posting_lists = std::mem::take(&mut builder.posting_lists);
                let tokens = std::mem::take(&mut builder.tokens);
                let tokens_batch = tokens.to_batch(token_set_format)?;
                let docs_batch = docs.to_batch()?;

                let writer_idx = (id as usize) % write_senders.len();
                let sender = &write_senders[writer_idx];
                send_write_op(
                    sender,
                    WriteOp::StartPartition {
                        id,
                        with_position,
                    },
                )?;
                for posting_list in posting_lists {
                    let block_max_scores = docs.calculate_block_max_scores(
                        posting_list.doc_ids.iter(),
                        posting_list.frequencies.iter(),
                    );
                    let batch = posting_list.to_batch(block_max_scores)?;
                    send_write_op(sender, WriteOp::PostingBatch { id, batch })?;
                }
                send_write_op(
                    sender,
                    WriteOp::FinishPartition {
                        id,
                        tokens_batch,
                        docs_batch,
                    },
                )?;
                Ok(id)
            };

            while let Ok(event) = read_receiver.recv_blocking() {
                match event {
                    MergeEvent::PartitionStart {
                        tokens,
                        docs,
                        inverted_list,
                    } => {
                        if (builder.docs.len() + docs.len() > u32::MAX as usize
                            || estimated_size >= target_size)
                            && !builder.tokens.is_empty()
                        {
                            let id = flush_builder(&mut builder)?;
                            partitions.push(id);
                            builder = InnerBuilder::new(id + 1, with_position, token_set_format);
                            estimated_size = builder.docs.size() + builder.tokens.estimated_size();
                        }

                        let old_tokens_size = builder.tokens.estimated_size();
                        let old_docs_size = builder.docs.size();

                        let tokens_len = tokens.len();
                        let mut token_map = vec![0u32; tokens_len];
                        for (token, token_id) in tokens.into_hash_map().into_iter() {
                            let new_id = builder.tokens.add(token);
                            token_map[token_id as usize] = new_id;
                        }

                        let doc_id_offset = builder.docs.len() as u32;
                        for (row_id, num_tokens) in docs.iter() {
                            builder.docs.append(*row_id, *num_tokens);
                        }

                        let new_tokens_size = builder.tokens.estimated_size();
                        let new_docs_size = builder.docs.size();
                        estimated_size += new_tokens_size - old_tokens_size;
                        estimated_size += new_docs_size - old_docs_size;

                        builder
                            .posting_lists
                            .resize_with(builder.tokens.len(), || {
                                PostingListBuilder::new(with_position)
                            });

                        current = Some(PartitionContext {
                            token_map,
                            doc_id_offset,
                            inverted_list,
                        });
                    }
                    MergeEvent::PostingBatch { token_id, batch } => {
                        let ctx = current.as_ref().ok_or_else(|| Error::Internal {
                            message: "posting batch received without active partition".to_owned(),
                            location: location!(),
                        })?;
                        let posting_list =
                            ctx.inverted_list.posting_list_from_batch(&batch, token_id)?;
                        let new_token_id = ctx.token_map[token_id as usize];
                        let list_builder = &mut builder.posting_lists[new_token_id as usize];
                        let old_size = list_builder.size();
                        for (doc_id, freq, positions) in posting_list.iter() {
                            let new_doc_id = ctx.doc_id_offset + doc_id as u32;
                            let positions = match positions {
                                Some(positions) => {
                                    PositionRecorder::Position(positions.collect())
                                }
                                None => PositionRecorder::Count(freq),
                            };
                            list_builder.add(new_doc_id, positions);
                        }
                        let new_size = list_builder.size();
                        estimated_size += new_size - old_size;
                    }
                    MergeEvent::PartitionEnd => {
                        current = None;
                        parts_merged += 1;
                        if parts_merged % 10 == 0 {
                            log::info!(
                                "merged {}/{} partitions in {:?}",
                                parts_merged,
                                num_parts,
                                start.elapsed()
                            );
                        }
                    }
                }
            }

            if !builder.tokens.is_empty() {
                let id = flush_builder(&mut builder)?;
                partitions.push(id);
            }
            Ok::<Vec<u64>, Error>(partitions)
        });

        let ((), partitions, _) = tokio::try_join!(reader, cpu, try_join_all(writer_futures))?;
        self.partitions = partitions;
        Ok(self.partitions.clone())
    }
}
