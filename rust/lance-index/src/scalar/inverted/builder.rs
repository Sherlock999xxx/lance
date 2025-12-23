// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use super::{
    index::*,
    merger::{Merger, SizeBasedMerger},
    InvertedIndexParams,
};
use crate::scalar::inverted::json::JsonTextStream;
use crate::scalar::inverted::lance_tokenizer::DocType;
use crate::scalar::inverted::tokenizer::lance_tokenizer::LanceTokenizer;
use crate::scalar::lance_format::LanceIndexStore;
use crate::scalar::IndexStore;
use crate::vector::graph::OrderedFloat;
use arrow::datatypes;
use arrow::array::AsArray;
use arrow_array::{Array, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use bitpacking::{BitPacker, BitPacker4x};
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream};
use deepsize::DeepSizeOf;
use futures::{stream, Stream, StreamExt, TryStreamExt};
use lance_arrow::json::JSON_EXT_NAME;
use lance_arrow::{iter_str_array, ARROW_EXT_NAME_KEY};
use lance_core::utils::tokio::get_num_compute_intensive_cpus;
use lance_core::{cache::LanceCache, utils::tokio::spawn_cpu};
use lance_core::{error::LanceOptionExt, utils::tempfile::TempDir};
use lance_core::{Error, Result, ROW_ID, ROW_ID_FIELD};
use lance_io::object_store::ObjectStore;
use object_store::path::Path;
use snafu::location;
use std::collections::HashMap;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::LazyLock;
use std::task::{Context, Poll};
use std::{fmt::Debug, sync::atomic::AtomicU64};
use tracing::instrument;

// the number of elements in each block
// each block contains 128 row ids and 128 frequencies
// WARNING: changing this value will break the compatibility with existing indexes
pub const BLOCK_SIZE: usize = BitPacker4x::BLOCK_LEN;

// the number of shards to split the indexing work,
// the indexing process would spawn `LANCE_FTS_NUM_SHARDS` workers to build FTS,
// higher for faster indexing performance, but more memory usage,
// it's `the number of compute intensive CPUs` by default
pub static LANCE_FTS_NUM_SHARDS: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_NUM_SHARDS")
        .unwrap_or_else(|_| get_num_compute_intensive_cpus().to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_NUM_SHARDS")
});
// the partition size limit in MiB (uncompressed format)
// higher for better indexing & query performance, but more memory usage,
pub static LANCE_FTS_PARTITION_SIZE: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_PARTITION_SIZE")
        .unwrap_or_else(|_| "256".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_PARTITION_SIZE")
});
// the target size of partition after merging in MiB (uncompressed format)
pub static LANCE_FTS_TARGET_SIZE: LazyLock<u64> = LazyLock::new(|| {
    std::env::var("LANCE_FTS_TARGET_SIZE")
        .unwrap_or_else(|_| "4096".to_string())
        .parse()
        .expect("failed to parse LANCE_FTS_TARGET_SIZE")
});

#[derive(Debug)]
pub struct InvertedIndexBuilder {
    params: InvertedIndexParams,
    pub(crate) partitions: Vec<u64>,
    new_partitions: Vec<u64>,
    fragment_mask: Option<u64>,
    token_set_format: TokenSetFormat,
    _tmpdir: TempDir,
    local_store: Arc<dyn IndexStore>,
    src_store: Arc<dyn IndexStore>,
}

impl InvertedIndexBuilder {
    pub fn new(params: InvertedIndexParams) -> Self {
        Self::new_with_fragment_mask(params, None)
    }

    pub fn new_with_fragment_mask(params: InvertedIndexParams, fragment_mask: Option<u64>) -> Self {
        Self::from_existing_index(
            params,
            None,
            Vec::new(),
            TokenSetFormat::default(),
            fragment_mask,
        )
    }

    /// Creates an InvertedIndexBuilder from existing index with fragment filtering.
    /// This method is used to create a builder from an existing index while applying
    /// fragment-based filtering for distributed indexing scenarios.
    /// fragment_mask Optional mask with fragment_id in high 32 bits for filtering.
    /// Constructed as `(fragment_id as u64) << 32`.
    /// When provided, ensures that generated IDs belong to the specified fragment.
    pub fn from_existing_index(
        params: InvertedIndexParams,
        store: Option<Arc<dyn IndexStore>>,
        partitions: Vec<u64>,
        token_set_format: TokenSetFormat,
        fragment_mask: Option<u64>,
    ) -> Self {
        let tmpdir = TempDir::default();
        let local_store = Arc::new(LanceIndexStore::new(
            ObjectStore::local().into(),
            tmpdir.obj_path(),
            Arc::new(LanceCache::no_cache()),
        ));
        let src_store = store.unwrap_or_else(|| local_store.clone());
        Self {
            params,
            partitions,
            new_partitions: Vec::new(),
            _tmpdir: tmpdir,
            local_store,
            src_store,
            token_set_format,
            fragment_mask,
        }
    }

    pub async fn update(
        &mut self,
        new_data: SendableRecordBatchStream,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        let schema = new_data.schema();
        let doc_col = schema.field(0).name();

        // infer lance_tokenizer based on document type
        if self.params.lance_tokenizer.is_none() {
            let schema = new_data.schema();
            let field = schema.column_with_name(doc_col).expect_ok()?.1;
            let doc_type = DocType::try_from(field)?;
            self.params.lance_tokenizer = Some(doc_type.as_ref().to_string());
        }

        let new_data = document_input(new_data, doc_col)?;

        self.update_index(new_data).await?;
        self.write(dest_store).await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn update_index(&mut self, stream: SendableRecordBatchStream) -> Result<()> {
        let num_workers = *LANCE_FTS_NUM_SHARDS;
        let tokenizer = self.params.build()?;
        let with_position = self.params.with_position;
        let next_id = self.partitions.iter().map(|id| id + 1).max().unwrap_or(0);
        let id_alloc = Arc::new(AtomicU64::new(next_id));
        let (batch_sender, batch_receiver) = async_channel::bounded(num_workers);
        let (partition_sender, partition_receiver) = async_channel::bounded(num_workers);

        let mut io_tasks = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let store = self.local_store.clone();
            let partition_receiver: async_channel::Receiver<PartitionBatches> =
                partition_receiver.clone();
            let task = tokio::task::spawn(async move {
                while let Ok(partition) = partition_receiver.recv().await {
                    partition.write(store.as_ref()).await?;
                }
                Result::Ok(())
            });
            io_tasks.push(task);
        }
        drop(partition_receiver);

        let mut index_tasks = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let tokenizer = tokenizer.clone();
            let batch_receiver: async_channel::Receiver<RecordBatch> = batch_receiver.clone();
            let partition_sender = partition_sender.clone();
            let id_alloc = id_alloc.clone();
            let fragment_mask = self.fragment_mask;
            let token_set_format = self.token_set_format;
            let task = spawn_cpu(move || {
                let mut worker = IndexWorker::new(
                    tokenizer,
                    with_position,
                    id_alloc,
                    fragment_mask,
                    token_set_format,
                )?;
                while let Ok(batch) = batch_receiver.recv_blocking() {
                    if u32::MAX - (worker.num_docs() as u32) < batch.num_rows() as u32 {
                        if let Some(partition) = worker.flush_into_batches()? {
                            partition_sender
                                .send_blocking(partition)
                                .map_err(|_| Error::Internal {
                                    message: "partition channel closed".to_owned(),
                                    location: location!(),
                                })?;
                        }
                    }
                    for partition in worker.process_batch(batch)? {
                        partition_sender
                            .send_blocking(partition)
                            .map_err(|_| Error::Internal {
                                message: "partition channel closed".to_owned(),
                                location: location!(),
                            })?;
                    }
                }
                if let Some(partition) = worker.flush_into_batches()? {
                    partition_sender
                        .send_blocking(partition)
                        .map_err(|_| Error::Internal {
                            message: "partition channel closed".to_owned(),
                            location: location!(),
                        })?;
                }
                Result::Ok(worker.partitions)
            });
            index_tasks.push(task);
        }

        drop(partition_sender);

        let mut stream = Box::pin(stream.then({
            |batch_result| {
                let batch_sender = batch_sender.clone();
                async move {
                    let batch = batch_result?;
                    let num_rows = batch.num_rows();
                    batch_sender
                        .send(batch)
                        .await
                        .map_err(|_| Error::Internal {
                            message: "batch channel closed".to_owned(),
                            location: location!(),
                        })?;
                    Result::Ok(num_rows)
                }
            }
        }));
        log::info!("indexing FTS with {} workers", num_workers);

        let mut last_num_rows = 0;
        let mut total_num_rows = 0;
        let start = std::time::Instant::now();
        while let Some(num_rows) = stream.try_next().await? {
            total_num_rows += num_rows;
            if total_num_rows >= last_num_rows + 1_000_000 {
                log::debug!(
                    "indexed {} documents, elapsed: {:?}, speed: {}rows/s",
                    total_num_rows,
                    start.elapsed(),
                    total_num_rows as f32 / start.elapsed().as_secs_f32()
                );
                last_num_rows = total_num_rows;
            }
        }
        // drop the sender to stop receivers
        drop(stream);
        drop(batch_sender);
        log::info!("dispatching elapsed: {:?}", start.elapsed());

        // wait for the workers to finish
        let start = std::time::Instant::now();
        for index_task in index_tasks {
            self.new_partitions.extend(index_task.await?);
        }
        for io_task in io_tasks {
            io_task.await??;
        }
        log::info!("wait workers indexing elapsed: {:?}", start.elapsed());
        Ok(())
    }

    pub async fn remap(
        &mut self,
        mapping: &HashMap<u64, Option<u64>>,
        src_store: Arc<dyn IndexStore>,
        dest_store: &dyn IndexStore,
    ) -> Result<()> {
        for part in self.partitions.iter() {
            let part = InvertedPartition::load(
                src_store.clone(),
                *part,
                None,
                &LanceCache::no_cache(),
                self.token_set_format,
            )
            .await?;
            let mut builder = part.into_builder().await?;
            builder.remap(mapping).await?;
            builder.write(dest_store).await?;
        }
        if self.fragment_mask.is_none() {
            self.write_metadata(dest_store, &self.partitions).await?;
        } else {
            // in distributed mode, the part_temp_metadata is written by the worker
            for &partition_id in &self.partitions {
                self.write_part_metadata(dest_store, partition_id).await?;
            }
        }
        Ok(())
    }

    async fn write_metadata(&self, dest_store: &dyn IndexStore, partitions: &[u64]) -> Result<()> {
        let metadata = HashMap::from_iter(vec![
            ("partitions".to_owned(), serde_json::to_string(&partitions)?),
            ("params".to_owned(), serde_json::to_string(&self.params)?),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                self.token_set_format.to_string(),
            ),
        ]);
        let mut writer = dest_store
            .new_index_file(METADATA_FILE, Arc::new(Schema::empty()))
            .await?;
        writer.finish_with_metadata(metadata).await?;
        Ok(())
    }

    /// Write partition metadata file for a single partition
    ///
    /// In a distributed environment, each worker node can write partition metadata files for the partitions it processes,
    /// which are then merged into a final metadata file using the `merge_metadata_files` function.
    pub(crate) async fn write_part_metadata(
        &self,
        dest_store: &dyn IndexStore,
        partition: u64, // Modify parameter type
    ) -> Result<()> {
        let partitions = vec![partition];
        let metadata = HashMap::from_iter(vec![
            ("partitions".to_owned(), serde_json::to_string(&partitions)?),
            ("params".to_owned(), serde_json::to_string(&self.params)?),
            (
                TOKEN_SET_FORMAT_KEY.to_owned(),
                self.token_set_format.to_string(),
            ),
        ]);
        // Use partition ID to generate a unique temporary filename
        let file_name = part_metadata_file_path(partition);
        let mut writer = dest_store
            .new_index_file(&file_name, Arc::new(Schema::empty()))
            .await?;
        writer.finish_with_metadata(metadata).await?;
        Ok(())
    }

    async fn write(&self, dest_store: &dyn IndexStore) -> Result<()> {
        let no_cache = LanceCache::no_cache();
        let partitions = futures::future::try_join_all(
            self.partitions
                .iter()
                .map(|part| {
                    InvertedPartition::load(
                        self.src_store.clone(),
                        *part,
                        None,
                        &no_cache,
                        self.token_set_format,
                    )
                })
                .chain(self.new_partitions.iter().map(|part| {
                    InvertedPartition::load(
                        self.local_store.clone(),
                        *part,
                        None,
                        &no_cache,
                        self.token_set_format,
                    )
                })),
        )
        .await?;
        let mut merger = SizeBasedMerger::new(
            dest_store,
            partitions,
            *LANCE_FTS_TARGET_SIZE << 20,
            self.token_set_format,
        );
        let partitions = merger.merge().await?;

        if self.fragment_mask.is_none() {
            self.write_metadata(dest_store, &partitions).await?;
        } else {
            for &partition_id in &partitions {
                self.write_part_metadata(dest_store, partition_id).await?;
            }
        }
        Ok(())
    }
}

impl Default for InvertedIndexBuilder {
    fn default() -> Self {
        let params = InvertedIndexParams::default();
        Self::new(params)
    }
}

// builder for single partition
#[derive(Debug)]
pub struct InnerBuilder {
    id: u64,
    with_position: bool,
    token_set_format: TokenSetFormat,
    pub(crate) tokens: TokenSet,
    pub(crate) posting_lists: Vec<PostingListBuilder>,
    pub(crate) docs: DocSet,
}

impl InnerBuilder {
    pub fn new(id: u64, with_position: bool, token_set_format: TokenSetFormat) -> Self {
        Self {
            id,
            with_position,
            token_set_format,
            tokens: TokenSet::default(),
            posting_lists: Vec::new(),
            docs: DocSet::default(),
        }
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    pub async fn remap(&mut self, mapping: &HashMap<u64, Option<u64>>) -> Result<()> {
        // for the docs, we need to remove the rows that are removed from the doc set,
        // and update the row ids of the rows that are updated
        let removed = self.docs.remap(mapping);

        // for the posting lists, we need to remap the doc ids:
        // - if the a row is removed, we need to shift the doc ids of the following rows
        // - if a row is updated (assigned a new row id), we don't need to do anything with the posting lists
        let mut token_id = 0;
        let mut removed_token_ids = Vec::new();
        self.posting_lists.retain_mut(|posting_list| {
            posting_list.remap(&removed);
            let keep = !posting_list.is_empty();
            if !keep {
                removed_token_ids.push(token_id as u32);
            }
            token_id += 1;
            keep
        });

        // for the tokens, remap the token ids if any posting list is empty
        self.tokens.remap(&removed_token_ids);

        Ok(())
    }

    pub async fn write(&mut self, store: &dyn IndexStore) -> Result<()> {
        let docs = Arc::new(std::mem::take(&mut self.docs));
        self.write_posting_lists(store, docs.clone()).await?;
        self.write_tokens(store).await?;
        self.write_docs(store, docs).await?;
        Ok(())
    }

    fn build_posting_list_batches(&mut self, docs: Arc<DocSet>) -> Result<Vec<RecordBatch>> {
        let posting_lists = std::mem::take(&mut self.posting_lists);
        let mut batches_out = Vec::new();
        for posting_list in posting_lists {
            let block_max_scores = docs.calculate_block_max_scores(
                posting_list.doc_ids.iter(),
                posting_list.frequencies.iter(),
            );
            let batch = posting_list.to_batch(block_max_scores)?;
            batches_out.push(batch);
        }
        Ok(batches_out)
    }

    pub(crate) fn build_partition_batches(&mut self) -> Result<PartitionBatches> {
        let docs = Arc::new(std::mem::take(&mut self.docs));
        let posting_batches = self.build_posting_list_batches(docs.clone())?;
        let tokens = std::mem::take(&mut self.tokens);
        let tokens_batch = tokens.to_batch(self.token_set_format)?;
        let docs_batch = docs.to_batch()?;
        Ok(PartitionBatches {
            id: self.id,
            with_position: self.with_position,
            posting_batches,
            tokens_batch,
            docs_batch,
        })
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_posting_lists(
        &mut self,
        store: &dyn IndexStore,
        docs: Arc<DocSet>,
    ) -> Result<()> {
        let id = self.id;
        let mut writer = store
            .new_index_file(
                &posting_file_path(self.id),
                inverted_list_schema(self.with_position),
            )
            .await?;
        let posting_lists = std::mem::take(&mut self.posting_lists);

        log::info!(
            "writing {} posting lists of partition {}, with position {}",
            posting_lists.len(),
            id,
            self.with_position
        );
        let mut batches = stream::iter(posting_lists)
            .map(|posting_list| {
                let block_max_scores = docs.calculate_block_max_scores(
                    posting_list.doc_ids.iter(),
                    posting_list.frequencies.iter(),
                );
                spawn_cpu(move || posting_list.to_batch(block_max_scores))
            })
            .buffered(get_num_compute_intensive_cpus());

        let mut write_duration = std::time::Duration::ZERO;
        let mut num_posting_lists = 0;
        while let Some(batch) = batches.try_next().await? {
            let start = std::time::Instant::now();
            writer.write_record_batch(batch).await?;
            write_duration += start.elapsed();
            num_posting_lists += 1;

            if num_posting_lists % 500_000 == 0 {
                log::info!(
                    "wrote {} posting lists of partition {}, writing elapsed: {:?}",
                    num_posting_lists,
                    id,
                    write_duration,
                );
            }
        }
        writer.finish().await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_tokens(&mut self, store: &dyn IndexStore) -> Result<()> {
        log::info!("writing tokens of partition {}", self.id);
        let tokens = std::mem::take(&mut self.tokens);
        let batch = tokens.to_batch(self.token_set_format)?;
        let mut writer = store
            .new_index_file(&token_file_path(self.id), batch.schema())
            .await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;
        Ok(())
    }

    #[instrument(level = "debug", skip_all)]
    async fn write_docs(&mut self, store: &dyn IndexStore, docs: Arc<DocSet>) -> Result<()> {
        log::info!("writing docs of partition {}", self.id);
        let batch = docs.to_batch()?;
        let mut writer = store
            .new_index_file(&doc_file_path(self.id), batch.schema())
            .await?;
        writer.write_record_batch(batch).await?;
        writer.finish().await?;
        Ok(())
    }
}

pub(crate) struct PartitionBatches {
    id: u64,
    with_position: bool,
    posting_batches: Vec<RecordBatch>,
    tokens_batch: RecordBatch,
    docs_batch: RecordBatch,
}

impl PartitionBatches {
    pub(crate) async fn write(self, store: &dyn IndexStore) -> Result<()> {
        let mut posting_writer = store
            .new_index_file(
                &posting_file_path(self.id),
                inverted_list_schema(self.with_position),
            )
            .await?;
        for batch in self.posting_batches {
            posting_writer.write_record_batch(batch).await?;
        }
        posting_writer.finish().await?;

        let mut token_writer = store
            .new_index_file(&token_file_path(self.id), self.tokens_batch.schema())
            .await?;
        token_writer.write_record_batch(self.tokens_batch).await?;
        token_writer.finish().await?;

        let mut doc_writer = store
            .new_index_file(&doc_file_path(self.id), self.docs_batch.schema())
            .await?;
        doc_writer.write_record_batch(self.docs_batch).await?;
        doc_writer.finish().await?;
        Ok(())
    }
}

struct IndexWorker {
    tokenizer: Box<dyn LanceTokenizer>,
    id_alloc: Arc<AtomicU64>,
    builder: InnerBuilder,
    partitions: Vec<u64>,
    schema: SchemaRef,
    estimated_size: u64,
    docs_size: u64,
    tokens_size: u64,
    total_doc_length: usize,
    fragment_mask: Option<u64>,
    token_set_format: TokenSetFormat,
    token_occurrences: HashMap<u32, PositionRecorder>,
}

impl IndexWorker {
    fn new(
        tokenizer: Box<dyn LanceTokenizer>,
        with_position: bool,
        id_alloc: Arc<AtomicU64>,
        fragment_mask: Option<u64>,
        token_set_format: TokenSetFormat,
    ) -> Result<Self> {
        let schema = inverted_list_schema(with_position);
        let builder = InnerBuilder::new(
            id_alloc.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                | fragment_mask.unwrap_or(0),
            with_position,
            token_set_format,
        );
        let docs_size = builder.docs.size();
        let tokens_size = builder.tokens.estimated_size();
        let estimated_size = docs_size + tokens_size;

        Ok(Self {
            tokenizer,
            builder,
            partitions: Vec::new(),
            id_alloc,
            schema,
            estimated_size,
            docs_size,
            tokens_size,
            total_doc_length: 0,
            fragment_mask,
            token_set_format,
            token_occurrences: HashMap::new(),
        })
    }

    fn has_position(&self) -> bool {
        self.schema.column_with_name(POSITION_COL).is_some()
    }

    fn num_docs(&self) -> usize {
        self.builder.docs.len()
    }

    fn next_builder(&self) -> InnerBuilder {
        InnerBuilder::new(
            self.id_alloc
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                | self.fragment_mask.unwrap_or(0),
            self.has_position(),
            self.token_set_format,
        )
    }

    fn flush_into_batches(&mut self) -> Result<Option<PartitionBatches>> {
        if self.builder.tokens.is_empty() {
            return Ok(None);
        }

        log::info!(
            "flushing posting lists, estimated size: {} MiB",
            self.estimated_size / (1024 * 1024)
        );
        let next_builder = self.next_builder();
        let mut builder = std::mem::replace(&mut self.builder, next_builder);
        let batches = builder.build_partition_batches()?;
        self.partitions.push(batches.id);
        self.docs_size = self.builder.docs.size();
        self.tokens_size = self.builder.tokens.estimated_size();
        self.estimated_size = self.docs_size + self.tokens_size;
        Ok(Some(batches))
    }

    fn process_batch(&mut self, batch: RecordBatch) -> Result<Vec<PartitionBatches>> {
        let mut flushed = Vec::new();
        let doc_col = batch.column(0);
        let doc_iter = iter_str_array(doc_col);
        let row_id_col = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();
        let docs = doc_iter
            .zip(row_id_col.values().iter())
            .filter_map(|(doc, row_id)| doc.map(|doc| (doc, *row_id)));

        let with_position = self.has_position();
        for (doc, row_id) in docs {
            self.token_occurrences.clear();
            let mut token_num = 0;
            {
                let mut token_stream = self.tokenizer.token_stream_for_doc(doc);
                while token_stream.advance() {
                    let token = token_stream.token_mut();
                    let token_text = std::mem::take(&mut token.text);
                    let token_id = self.builder.tokens.add(token_text) as usize;
                    self.token_occurrences
                        .entry(token_id as u32)
                        .or_insert_with(|| PositionRecorder::new(with_position))
                        .push(token.position as u32);
                    token_num += 1;
                }
            }
            let new_tokens_size = self.builder.tokens.estimated_size();
            self.estimated_size += new_tokens_size - self.tokens_size;
            self.tokens_size = new_tokens_size;
            self.builder
                .posting_lists
                .resize_with(self.builder.tokens.len(), || {
                    PostingListBuilder::new(with_position)
                });
            let doc_id = self.builder.docs.append(row_id, token_num);
            let new_docs_size = self.builder.docs.size();
            self.estimated_size += new_docs_size - self.docs_size;
            self.docs_size = new_docs_size;
            self.total_doc_length += doc.len();

            for (token_id, term_positions) in self.token_occurrences.drain() {
                let posting_list = &mut self.builder.posting_lists[token_id as usize];

                let old_size = posting_list.size();
                posting_list.add(doc_id, term_positions);
                let new_size = posting_list.size();
                self.estimated_size += new_size - old_size;
            }

            if self.estimated_size >= *LANCE_FTS_PARTITION_SIZE << 20 {
                if let Some(batches) = self.flush_into_batches()? {
                    flushed.push(batches);
                }
            }
        }

        Ok(flushed)
    }
}

#[derive(Debug, Clone)]
pub enum PositionRecorder {
    Position(Vec<u32>),
    Count(u32),
}

impl PositionRecorder {
    fn new(with_position: bool) -> Self {
        if with_position {
            Self::Position(Vec::new())
        } else {
            Self::Count(0)
        }
    }

    fn push(&mut self, position: u32) {
        match self {
            Self::Position(positions) => positions.push(position),
            Self::Count(count) => *count += 1,
        }
    }

    pub fn len(&self) -> u32 {
        match self {
            Self::Position(positions) => positions.len() as u32,
            Self::Count(count) => *count,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn into_vec(self) -> Vec<u32> {
        match self {
            Self::Position(positions) => positions,
            Self::Count(_) => vec![0],
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, DeepSizeOf)]
pub struct ScoredDoc {
    pub row_id: u64,
    pub score: OrderedFloat,
}

impl ScoredDoc {
    pub fn new(row_id: u64, score: f32) -> Self {
        Self {
            row_id,
            score: OrderedFloat(score),
        }
    }
}

impl PartialOrd for ScoredDoc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredDoc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.cmp(&other.score)
    }
}

pub fn legacy_inverted_list_schema(with_position: bool) -> SchemaRef {
    let mut fields = vec![
        arrow_schema::Field::new(ROW_ID, arrow_schema::DataType::UInt64, false),
        arrow_schema::Field::new(FREQUENCY_COL, arrow_schema::DataType::Float32, false),
    ];
    if with_position {
        fields.push(arrow_schema::Field::new(
            POSITION_COL,
            arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::Int32,
                true,
            ))),
            false,
        ));
    }
    Arc::new(arrow_schema::Schema::new(fields))
}

pub fn inverted_list_schema(with_position: bool) -> SchemaRef {
    let mut fields = vec![
        // we compress the posting lists (including row ids and frequencies),
        // and store the compressed posting lists, so it's a large binary array
        arrow_schema::Field::new(
            POSTING_COL,
            datatypes::DataType::List(Arc::new(Field::new(
                "item",
                datatypes::DataType::LargeBinary,
                true,
            ))),
            false,
        ),
        arrow_schema::Field::new(MAX_SCORE_COL, datatypes::DataType::Float32, false),
        arrow_schema::Field::new(LENGTH_COL, datatypes::DataType::UInt32, false),
    ];
    if with_position {
        fields.push(arrow_schema::Field::new(
            POSITION_COL,
            arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                "item",
                arrow_schema::DataType::List(Arc::new(arrow_schema::Field::new(
                    "item",
                    arrow_schema::DataType::LargeBinary,
                    true,
                ))),
                true,
            ))),
            false,
        ));
    }
    Arc::new(arrow_schema::Schema::new(fields))
}

/// Flatten the string list stream into a string stream
pub struct FlattenStream {
    /// Inner record batch stream with 2 columns:
    /// 1. doc_col: List(Utf8) or List(LargeUtf8)
    /// 2. row_id_col: UInt64
    inner: SendableRecordBatchStream,
    field_type: DataType,
    data_type: DataType,
}

impl FlattenStream {
    pub fn new(input: SendableRecordBatchStream) -> Self {
        let schema = input.schema();
        let field = schema.field(0);
        let data_type = match field.data_type() {
            DataType::List(f) if matches!(f.data_type(), DataType::Utf8) => DataType::Utf8,
            DataType::List(f) if matches!(f.data_type(), DataType::LargeUtf8) => {
                DataType::LargeUtf8
            }
            DataType::LargeList(f) if matches!(f.data_type(), DataType::Utf8) => DataType::Utf8,
            DataType::LargeList(f) if matches!(f.data_type(), DataType::LargeUtf8) => {
                DataType::LargeUtf8
            }
            _ => panic!(
                "expect data type List(Utf8) or List(LargeUtf8) but got {:?}",
                field.data_type()
            ),
        };
        Self {
            inner: input,
            field_type: field.data_type().clone(),
            data_type,
        }
    }
}

impl Stream for FlattenStream {
    type Item = datafusion_common::Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(batch))) => {
                let doc_col = batch.column(0);
                let batch = match self.field_type {
                    DataType::List(_) => flatten_string_list::<i32>(&batch, doc_col).map_err(|e| {
                        datafusion_common::error::DataFusionError::Execution(format!(
                            "flatten string list error: {}",
                            e
                        ))
                    }),
                    DataType::LargeList(_) => {
                        flatten_string_list::<i64>(&batch, doc_col).map_err(|e| {
                            datafusion_common::error::DataFusionError::Execution(format!(
                                "flatten string list error: {}",
                                e
                            ))
                        })
                    }
                    _ => unreachable!(
                        "expect data type List or LargeList but got {:?}",
                        self.field_type
                    ),
                };
                Poll::Ready(Some(batch))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl RecordBatchStream for FlattenStream {
    fn schema(&self) -> SchemaRef {
        let schema = Schema::new(vec![
            Field::new(
                self.inner.schema().field(0).name(),
                self.data_type.clone(),
                true,
            ),
            ROW_ID_FIELD.clone(),
        ]);

        Arc::new(schema)
    }
}

fn flatten_string_list<Offset: arrow::array::OffsetSizeTrait>(
    batch: &RecordBatch,
    doc_col: &Arc<dyn Array>,
) -> Result<RecordBatch> {
    let docs = doc_col.as_list::<Offset>();
    let row_ids = batch[ROW_ID].as_primitive::<datatypes::UInt64Type>();

    let row_ids = row_ids
        .values()
        .iter()
        .zip(docs.iter())
        .flat_map(|(row_id, doc)| std::iter::repeat_n(*row_id, doc.map(|d| d.len()).unwrap_or(0)));

    let row_ids = Arc::new(UInt64Array::from_iter_values(row_ids));
    let docs = match docs.value_type() {
        datatypes::DataType::Utf8 | datatypes::DataType::LargeUtf8 => docs.values().clone(),
        _ => {
            return Err(Error::Index {
                message: format!(
                    "expect data type String or LargeString but got {}",
                    docs.value_type()
                ),
                location: location!(),
            });
        }
    };

    let schema = Schema::new(vec![
        Field::new(
            batch.schema().field(0).name(),
            docs.data_type().clone(),
            true,
        ),
        ROW_ID_FIELD.clone(),
    ]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![docs, row_ids])?;
    Ok(batch)
}

pub(crate) fn token_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, TOKENS_FILE)
}

pub(crate) fn posting_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, INVERT_LIST_FILE)
}

pub(crate) fn doc_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, DOCS_FILE)
}

pub(crate) fn part_metadata_file_path(partition_id: u64) -> String {
    format!("part_{}_{}", partition_id, METADATA_FILE)
}

pub async fn merge_index_files(
    object_store: &ObjectStore,
    index_dir: &Path,
    store: Arc<dyn IndexStore>,
) -> Result<()> {
    // List all partition metadata files in the index directory
    let part_metadata_files = list_metadata_files(object_store, index_dir).await?;

    // Call merge_metadata_files function for inverted index
    merge_metadata_files(store, &part_metadata_files).await
}

/// List and filter metadata files from the index directory
/// Returns partition metadata files
async fn list_metadata_files(object_store: &ObjectStore, index_dir: &Path) -> Result<Vec<String>> {
    // List all partition metadata files in the index directory
    let mut part_metadata_files = Vec::new();
    let mut list_stream = object_store.list(Some(index_dir.clone()));

    while let Some(item) = list_stream.next().await {
        match item {
            Ok(meta) => {
                let file_name = meta.location.filename().unwrap_or_default();
                // Filter files matching the pattern part_*_metadata.lance
                if file_name.starts_with("part_") && file_name.ends_with("_metadata.lance") {
                    part_metadata_files.push(file_name.to_string());
                }
            }
            Err(_) => continue,
        }
    }

    if part_metadata_files.is_empty() {
        return Err(Error::InvalidInput {
            source: format!(
                "No partition metadata files found in index directory: {}",
                index_dir
            )
            .into(),
            location: location!(),
        });
    }

    Ok(part_metadata_files)
}

/// Merge partition metadata files with partition ID remapping to sequential IDs starting from 0
async fn merge_metadata_files(
    store: Arc<dyn IndexStore>,
    part_metadata_files: &[String],
) -> Result<()> {
    // Collect all partition IDs and params
    let mut all_partitions = Vec::new();
    let mut params = None;
    let mut token_set_format = None;

    for file_name in part_metadata_files {
        let reader = store.open_index_file(file_name).await?;
        let metadata = &reader.schema().metadata;

        let partitions_str = metadata.get("partitions").ok_or(Error::Index {
            message: format!("partitions not found in {}", file_name),
            location: location!(),
        })?;

        let partition_ids: Vec<u64> =
            serde_json::from_str(partitions_str).map_err(|e| Error::Index {
                message: format!("Failed to parse partitions: {}", e),
                location: location!(),
            })?;

        all_partitions.extend(partition_ids);

        if params.is_none() {
            let params_str = metadata.get("params").ok_or(Error::Index {
                message: format!("params not found in {}", file_name),
                location: location!(),
            })?;
            params = Some(
                serde_json::from_str::<InvertedIndexParams>(params_str).map_err(|e| {
                    Error::Index {
                        message: format!("Failed to parse params: {}", e),
                        location: location!(),
                    }
                })?,
            );
        }

        if token_set_format.is_none() {
            if let Some(name) = metadata.get(TOKEN_SET_FORMAT_KEY) {
                token_set_format = Some(TokenSetFormat::from_str(name)?);
            }
        }
    }

    // Create ID mapping: sorted original IDs -> 0,1,2...
    let mut sorted_ids = all_partitions.clone();
    sorted_ids.sort();
    sorted_ids.dedup();

    let id_mapping: HashMap<u64, u64> = sorted_ids
        .iter()
        .enumerate()
        .map(|(new_id, &old_id)| (old_id, new_id as u64))
        .collect();

    // Safe rename partition files using temporary files to avoid overwrite
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    // Phase 1: Move files to temporary locations
    let mut temp_files: Vec<(String, String, String)> = Vec::new(); // (temp_path, old_path, final_path)

    for (&old_id, &new_id) in &id_mapping {
        if old_id != new_id {
            for suffix in [TOKENS_FILE, INVERT_LIST_FILE, DOCS_FILE] {
                let old_path = format!("part_{}_{}", old_id, suffix);
                let new_path = format!("part_{}_{}", new_id, suffix);
                let temp_path = format!("temp_{}_{}", timestamp, old_path);

                // Move to temporary location first to avoid overwrite
                if let Err(e) = store.rename_index_file(&old_path, &temp_path).await {
                    // Rollback phase 1: restore files from temp locations
                    for (temp_name, old_name, _) in temp_files.iter().rev() {
                        let _ = store.rename_index_file(temp_name, old_name).await;
                    }
                    return Err(Error::Index {
                        message: format!(
                            "Failed to move {} to temp {}: {}",
                            old_path, temp_path, e
                        ),
                        location: location!(),
                    });
                }
                temp_files.push((temp_path, old_path, new_path));
            }
        }
    }

    // Phase 2: Move from temporary to final locations
    let mut completed_renames: Vec<(String, String)> = Vec::new(); // (final_path, temp_path)

    for (temp_path, _old_path, final_path) in &temp_files {
        if let Err(e) = store.rename_index_file(temp_path, final_path).await {
            // Rollback phase 2: restore completed renames and remaining temps
            for (final_name, temp_name) in completed_renames.iter().rev() {
                let _ = store.rename_index_file(final_name, temp_name).await;
            }
            // Restore remaining temp files to original locations
            for (temp_name, orig_name, _) in temp_files.iter() {
                if !completed_renames.iter().any(|(_, t)| t == temp_name) {
                    let _ = store.rename_index_file(temp_name, orig_name).await;
                }
            }
            return Err(Error::Index {
                message: format!("Failed to rename {} to {}: {}", temp_path, final_path, e),
                location: location!(),
            });
        }
        completed_renames.push((final_path.clone(), temp_path.clone()));
    }

    // Write merged metadata with remapped IDs
    let remapped_partitions: Vec<u64> = (0..id_mapping.len() as u64).collect();
    let params = params.unwrap_or_default();
    let token_set_format = token_set_format.unwrap_or(TokenSetFormat::Arrow);
    let builder = InvertedIndexBuilder::from_existing_index(
        params,
        None,
        remapped_partitions.clone(),
        token_set_format,
        None,
    );
    builder
        .write_metadata(&*store, &remapped_partitions)
        .await?;

    // Cleanup partition metadata files
    for file_name in part_metadata_files {
        if file_name.starts_with("part_") && file_name.ends_with("_metadata.lance") {
            let _ = store.delete_index_file(file_name).await;
        }
    }

    Ok(())
}

/// Convert input stream into a stream of documents.
///
/// The input stream must be one of:
/// 1. Document in Utf8 or LargeUtf8 format.
/// 2. Document in List(Utf8) or List(LargeUtf8) format.
/// 3. Json document in LargeBinary format.
pub fn document_input(
    input: SendableRecordBatchStream,
    column: &str,
) -> Result<SendableRecordBatchStream> {
    let schema = input.schema();
    let field = schema.column_with_name(column).expect_ok()?.1;
    match field.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 => Ok(input),
        DataType::List(field) | DataType::LargeList(field)
            if matches!(field.data_type(), DataType::Utf8 | DataType::LargeUtf8) =>
        {
            Ok(Box::pin(FlattenStream::new(input)))
        }
        DataType::LargeBinary => match field.metadata().get(ARROW_EXT_NAME_KEY) {
            Some(name) if name.as_str() == JSON_EXT_NAME => {
                Ok(Box::pin(JsonTextStream::new(input, column.to_string())))
            }
            _ => Err(Error::InvalidInput {
                source: format!("column {} is not json", column).into(),
                location: location!(),
            }),
        },
        _ => Err(Error::InvalidInput {
            source: format!(
                "column {} has type {}, is not utf8, large utf8 type/list, or large binary",
                column,
                field.data_type()
            )
            .into(),
            location: location!(),
        }),
    }
}
