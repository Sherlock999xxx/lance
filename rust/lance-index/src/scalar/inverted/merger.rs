// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::RecordBatch;
use futures::future::try_join_all;
use lance_core::utils::tokio::spawn_cpu;
use lance_core::{Error, Result};
use snafu::location;
use std::sync::Arc;

use crate::scalar::IndexStore;

use super::{
    builder::{
        doc_file_path, posting_file_path, token_file_path, InnerBuilder, PartitionBatches,
        PositionRecorder,
    },
    DocSet, InvertedPartition, PostingListBuilder, PostingListReader, TokenSet, TokenSetFormat,
};

struct MergeInput {
    tokens: TokenSet,
    docs: DocSet,
    inverted_list: Arc<PostingListReader>,
    postings: RecordBatch,
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
        let (read_sender, read_receiver) = async_channel::bounded(num_workers);
        let (partition_sender, partition_receiver) = async_channel::bounded(num_workers);

        let mut writer_futures = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let receiver: async_channel::Receiver<PartitionBatches> = partition_receiver.clone();
            let store = self.dest_store;
            writer_futures.push(async move {
                while let Ok(partition) = receiver.recv().await {
                    partition.write(store).await?;
                }
                Result::Ok(())
            });
        }
        drop(partition_receiver);

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
                let has_positions = inverted_list.has_positions();
                let postings = inverted_list.read_batch(has_positions).await?;
                read_sender
                    .send(MergeInput {
                        tokens,
                        docs,
                        inverted_list,
                        postings,
                    })
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

            while let Ok(input) = read_receiver.recv_blocking() {
                if (builder.docs.len() + input.docs.len() > u32::MAX as usize
                    || estimated_size >= target_size)
                    && !builder.tokens.is_empty()
                {
                    let id = builder.id();
                    let batches = builder.build_partition_batches()?;
                    partition_sender
                        .send_blocking(batches)
                        .map_err(|_| Error::Internal {
                            message: "partition channel closed".to_owned(),
                            location: location!(),
                        })?;
                    partitions.push(id);
                    builder = InnerBuilder::new(id + 1, with_position, token_set_format);
                    estimated_size = builder.docs.size() + builder.tokens.estimated_size();
                }

                let old_tokens_size = builder.tokens.estimated_size();
                let old_docs_size = builder.docs.size();

                let mut token_map = vec![0u32; input.tokens.len()];
                for (token, token_id) in input.tokens.iter() {
                    let new_id = builder.tokens.add(token);
                    token_map[token_id as usize] = new_id;
                }

                let doc_id_offset = builder.docs.len() as u32;
                for (row_id, num_tokens) in input.docs.iter() {
                    builder.docs.append(*row_id, *num_tokens);
                }

                let new_tokens_size = builder.tokens.estimated_size();
                let new_docs_size = builder.docs.size();
                estimated_size += new_tokens_size - old_tokens_size;
                estimated_size += new_docs_size - old_docs_size;

                builder
                    .posting_lists
                    .resize_with(builder.tokens.len(), || PostingListBuilder::new(with_position));

                for token_id in 0..input.tokens.len() as u32 {
                    let range = input.inverted_list.posting_list_range(token_id);
                    let posting_list_batch =
                        input.postings.slice(range.start, range.end - range.start);
                    let posting_list =
                        input
                            .inverted_list
                            .posting_list_from_batch(&posting_list_batch, token_id)?;
                    let new_token_id = token_map[token_id as usize];
                    let list_builder = &mut builder.posting_lists[new_token_id as usize];
                    let old_size = list_builder.size();
                    for (doc_id, freq, positions) in posting_list.iter() {
                        let new_doc_id = doc_id_offset + doc_id as u32;
                        let positions = match positions {
                            Some(positions) => PositionRecorder::Position(positions.collect()),
                            None => PositionRecorder::Count(freq),
                        };
                        list_builder.add(new_doc_id, positions);
                    }
                    let new_size = list_builder.size();
                    estimated_size += new_size - old_size;
                }

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

            if !builder.tokens.is_empty() {
                let id = builder.id();
                let batches = builder.build_partition_batches()?;
                partition_sender
                    .send_blocking(batches)
                    .map_err(|_| Error::Internal {
                        message: "partition channel closed".to_owned(),
                        location: location!(),
                    })?;
                partitions.push(id);
            }
            Ok::<Vec<u64>, Error>(partitions)
        });

        let ((), partitions, _) = tokio::try_join!(reader, cpu, try_join_all(writer_futures))?;
        self.partitions = partitions;
        Ok(self.partitions.clone())
    }
}
