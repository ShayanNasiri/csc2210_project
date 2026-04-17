# MS MARCO Passage Ranking Data

## Source

- **Dataset**: MS MARCO Passage Ranking v1
- **Loaded via**: `ir_datasets` (`msmarco-passage/train/judged` and `msmarco-passage/dev/small`), with HuggingFace `datasets` (`ms_marco`, `v1.1`) as fallback

## Splits

### Training (`msmarco_train.parquet`)

- **Queries**: ~9,971 (randomly sampled from training set, `seed=42`; some sampled queries have no scored docs)
- **Candidates per query**: up to 100 (top BM25 candidates; not all queries have 100)
- **Total pairs**: ~976,698
- **Positive labels**: ~1,119

### Dev (`msmarco_dev.parquet`)

- **Queries**: 744 (from official dev/small set, filtered to only queries with at least one relevant candidate in top-100 BM25 list)
- **Candidates per query**: up to 100 (not all queries have 100)
- **Total pairs**: ~62,357
- **Positive labels**: ~748

### Train/Val Split (`msmarco_train_split.parquet`, `msmarco_val.parquet`)

Produced by `data/split_train_val.py`. The original training set is deterministically partitioned by query ID (`seed=42`) into a 95% training split (used for off-ramp and joint training) and a 5% validation split (used for hyperparameter tuning of α, β, and per-ramp threshold vectors). The dev set is reserved as the final test set.

- **`msmarco_train_split.parquet`**: ~95% of training queries (~9,472), pairs retained
- **`msmarco_val.parquet`**: ~5% of training queries (~499), pairs retained
- **No query overlap** between the two splits

## Pre-tokenized Dev Set (`dev_tokenized.pt`)

- **Tokenizer**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Max length**: 512 (padding="max_length", truncation=True)
- **Contents**: `{"input_ids": Tensor(62357, 512), "attention_mask": Tensor(62357, 512), "token_type_ids": Tensor(62357, 512), "qids": List[str], "labels": List[int]}`
- **Purpose**: Eliminates CPU tokenization overhead during inference latency measurement

## Pre-tokenized Validation Set (`val_tokenized.pt`)

Produced by `data/split_train_val.py` alongside the val parquet, with the same tokenization settings as `dev_tokenized.pt`. Used for hyperparameter sweeps (System D alpha, System E beta, System F/G/H per-ramp thresholds).

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
# Step 1: download raw parquets + tokenize dev set
python data/download_data.py                    # Linux / cluster (default UTF-8)
PYTHONUTF8=1 python data/download_data.py       # Windows

# Step 2: carve a 5% validation split from training and tokenize it
python data/split_train_val.py
```

All data files are excluded from git via `.gitignore` and must be regenerated on each machine.
