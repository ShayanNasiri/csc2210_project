# MS MARCO Passage Ranking Data

## Source

- **Dataset**: MS MARCO Passage Ranking v1
- **Loaded via**: `ir_datasets` (`msmarco-passage/train/judged` and `msmarco-passage/dev/small`), with HuggingFace `datasets` (`ms_marco`, `v1.1`) as fallback

## Splits

### Training (`msmarco_train.parquet`)

- **Queries**: 10,000 (randomly sampled from training set, `seed=42`)
- **Candidates per query**: up to 100 (top BM25 candidates)
- **Total pairs**: ~1,000,000
- **Positive labels**: ~10,000

### Dev (`msmarco_dev.parquet`)

- **Queries**: 1,000 (first 1K in sorted qid order from official dev/small set)
- **Candidates per query**: up to 100
- **Total pairs**: 100,000
- **Positive labels**: ~1,000

## Pre-tokenized Dev Set (`dev_tokenized.pt`)

- **Tokenizer**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Max length**: 512 (padding="max_length", truncation=True)
- **Contents**: `{"input_ids": Tensor(100000, 512), "attention_mask": Tensor(100000, 512), "qids": List[str], "labels": List[int]}`
- **Purpose**: Eliminates CPU tokenization overhead during inference latency measurement

## Schema

All Parquet files share the same columns:

| Column   | Type | Description |
|----------|------|-------------|
| `qid`    | str  | Query ID |
| `docid`  | str  | Document/passage ID |
| `query`  | str  | Query text |
| `passage`| str  | Passage text |
| `label`  | int  | Relevance label (0 = not relevant, 1 = relevant) |

## Regeneration

```bash
# Linux / cluster (default UTF-8)
python data/download_data.py

# Windows (must set UTF-8 mode for ir_datasets corpus)
PYTHONUTF8=1 python data/download_data.py
```

All data files are excluded from git via `.gitignore` and must be regenerated on each machine.
