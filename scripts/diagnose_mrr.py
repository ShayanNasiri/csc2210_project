"""Quick diagnostic to understand low MRR@10 in Baseline A."""

import pandas as pd
import torch

# Check dev data stats
dev = pd.read_parquet("data/msmarco_dev.parquet")
total_q = dev["qid"].nunique()
q_with_pos = dev.groupby("qid")["label"].sum()
print(f"Total queries: {total_q}")
print(f"Queries with at least 1 positive: {(q_with_pos > 0).sum()}")
print(f"Queries with 0 positives: {(q_with_pos == 0).sum()}")
print(f"Avg candidates per query: {len(dev) / total_q:.1f}")
print(f"Total pairs: {len(dev)}")
print(f"Total positives: {dev['label'].sum()}")

# Check tokenized data alignment
t = torch.load("data/dev_tokenized.pt", weights_only=False)
print(f"\nTokenized samples: {t['input_ids'].shape[0]}")
print(f"Tokenized qids count: {len(t['qids'])}")
print(f"Tokenized labels count: {len(t['labels'])}")
print(f"Tokenized positives: {sum(t['labels'])}")
print(f"Has token_type_ids: {'token_type_ids' in t}")

# Check first few qids match
print(f"\nFirst 5 parquet qids: {dev['qid'].tolist()[:5]}")
print(f"First 5 tokenized qids: {t['qids'][:5]}")
print(f"First 5 parquet labels: {dev['label'].tolist()[:5]}")
print(f"First 5 tokenized labels: {t['labels'][:5]}")
