"""Train off-ramp classifiers with frozen backbone."""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

from src.constants import MODEL_NAME, MAX_TOKEN_LENGTH
from src.model import EarlyExitCrossEncoder
from src.utils import get_device, set_seed


def tokenize_training_data(data_path: str, batch_size: int = 10_000):
    """Tokenize training pairs from parquet file."""
    print(f"Loading training data from {data_path} ...")
    df = pd.read_parquet(data_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    queries = df["query"].tolist()
    passages = df["passage"].tolist()
    labels = df["label"].tolist()

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    for start in tqdm(range(0, len(queries), batch_size), desc="Tokenizing"):
        end = min(start + batch_size, len(queries))
        encoded = tokenizer(
            queries[start:end],
            passages[start:end],
            max_length=MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_mask.append(encoded["attention_mask"])
        all_token_type_ids.append(encoded["token_type_ids"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    token_type_ids = torch.cat(all_token_type_ids, dim=0)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return input_ids, attention_mask, token_type_ids, labels_tensor


def train_offramps(
    data_path: str = "data/msmarco_train.parquet",
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    output_dir: str = "results",
    max_steps: int = -1,
):
    """Train off-ramp heads with frozen backbone."""
    set_seed()
    device = get_device()

    # Tokenize training data
    input_ids, attention_mask, token_type_ids, labels = tokenize_training_data(data_path)
    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model
    print("Loading model ...")
    model = EarlyExitCrossEncoder(MODEL_NAME)
    model.to(device)
    model.train()

    # Only optimize off-ramp parameters
    optimizer = torch.optim.AdamW(model.offramps.parameters(), lr=lr)

    global_step = 0
    for epoch in range(epochs):
        running_losses = [0.0] * 5
        step_count = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            b_input_ids, b_attention_mask, b_token_type_ids, b_labels = [
                x.to(device) for x in batch
            ]

            out = model.forward_with_offramps(
                b_input_ids, b_attention_mask, b_token_type_ids
            )

            # BCE loss for each off-ramp
            losses = []
            for i, logit in enumerate(out["offramp_logits"]):
                loss = F.binary_cross_entropy_with_logits(logit, b_labels)
                losses.append(loss)
                running_losses[i] += loss.item()

            total_loss = sum(losses) / len(losses)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            step_count += 1
            global_step += 1

            if global_step % 500 == 0:
                avg = [rl / step_count for rl in running_losses]
                print(f"  Step {global_step} — per-ramp avg loss: "
                      f"[{', '.join(f'{l:.4f}' for l in avg)}]")

            if max_steps > 0 and global_step >= max_steps:
                break

        # Epoch summary
        avg = [rl / step_count for rl in running_losses]
        print(f"Epoch {epoch + 1} — per-ramp avg loss: "
              f"[{', '.join(f'{l:.4f}' for l in avg)}]")

        if max_steps > 0 and global_step >= max_steps:
            break

    # Save off-ramp weights
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "offramp_weights.pt")
    torch.save(model.offramps.state_dict(), save_path)
    print(f"Saved off-ramp weights to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/msmarco_train.parquet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    train_offramps(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
