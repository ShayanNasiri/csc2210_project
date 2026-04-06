"""System D: Joint backbone + off-ramp training with differential learning rates.

Unlike train_offramps.py (which freezes the backbone), this script unfreezes all
backbone parameters and trains them jointly with the off-ramp heads. The combined
loss includes both the final classifier loss and the average off-ramp loss, weighted
by alpha. This forces early BERT layers to develop discriminative features for the
off-ramps, improving early-exit accuracy.

Existing weights (results/offramp_weights.pt) are NOT modified. Joint-trained
weights are saved to results/joint_weights.pt as a dict with keys:
  - "backbone": full backbone state_dict
  - "offramps": off-ramp collection state_dict
"""

import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.constants import MODEL_NAME, MAX_TOKEN_LENGTH, DEFAULT_JOINT_WEIGHTS_PATH
from src.model import EarlyExitCrossEncoder
from src.train_offramps import tokenize_training_data
from src.utils import get_device, set_seed


def train_joint(
    data_path: str = "data/msmarco_train.parquet",
    epochs: int = 3,
    batch_size: int = 64,
    backbone_lr: float = 2e-5,
    offramp_lr: float = 1e-3,
    alpha: float = 1.0,
    output_dir: str = "results",
    max_steps: int = -1,
):
    """Train backbone + off-ramps jointly with combined loss.

    Args:
        data_path: Path to training parquet file.
        epochs: Number of training epochs.
        batch_size: Batch size (smaller than offramp-only training due to
                    backbone gradients using more memory).
        backbone_lr: Learning rate for backbone parameters.
        offramp_lr: Learning rate for off-ramp parameters.
        alpha: Weight for off-ramp losses relative to final classifier loss.
               total_loss = loss_final + alpha * mean(offramp_losses)
        output_dir: Directory to save joint_weights.pt.
        max_steps: Max training steps (-1 for full training).
    """
    set_seed()
    device = get_device()

    # Tokenize training data (reuses the same function from train_offramps)
    input_ids, attention_mask, token_type_ids, labels = tokenize_training_data(data_path)
    dataset = TensorDataset(input_ids, attention_mask, token_type_ids, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model
    print("Loading model ...")
    model = EarlyExitCrossEncoder(MODEL_NAME)

    # Unfreeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = True

    model.to(device)
    model.train()

    # Differential learning rates
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": backbone_lr},
        {"params": model.offramps.parameters(), "lr": offramp_lr},
    ])

    print(f"Training jointly: backbone_lr={backbone_lr}, offramp_lr={offramp_lr}, alpha={alpha}")
    print(f"Trainable params: backbone={sum(p.numel() for p in model.backbone.parameters()):,}, "
          f"offramps={sum(p.numel() for p in model.offramps.parameters()):,}")

    global_step = 0
    for epoch in range(epochs):
        running_final_loss = 0.0
        running_ramp_losses = [0.0] * 5
        step_count = 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            b_input_ids, b_attention_mask, b_token_type_ids, b_labels = [
                x.to(device) for x in batch
            ]

            out = model.forward_with_offramps(
                b_input_ids, b_attention_mask, b_token_type_ids
            )

            # Final classifier loss
            final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], b_labels)

            # Off-ramp losses
            ramp_losses = []
            for i, logit in enumerate(out["offramp_logits"]):
                loss = F.binary_cross_entropy_with_logits(logit, b_labels)
                ramp_losses.append(loss)
                running_ramp_losses[i] += loss.item()

            # Combined loss
            total_loss = final_loss + alpha * sum(ramp_losses) / len(ramp_losses)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_final_loss += final_loss.item()
            step_count += 1
            global_step += 1

            if global_step % 500 == 0:
                avg_final = running_final_loss / step_count
                avg_ramps = [rl / step_count for rl in running_ramp_losses]
                print(f"  Step {global_step} — final_loss: {avg_final:.4f}, "
                      f"ramp_losses: [{', '.join(f'{l:.4f}' for l in avg_ramps)}]")

            if max_steps > 0 and global_step >= max_steps:
                break

        # Epoch summary
        avg_final = running_final_loss / step_count
        avg_ramps = [rl / step_count for rl in running_ramp_losses]
        print(f"Epoch {epoch + 1} — final_loss: {avg_final:.4f}, "
              f"ramp_losses: [{', '.join(f'{l:.4f}' for l in avg_ramps)}]")

        if max_steps > 0 and global_step >= max_steps:
            break

    # Save full model state (backbone + offramps)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, DEFAULT_JOINT_WEIGHTS_PATH.split("/")[-1])
    state = {
        "backbone": model.backbone.state_dict(),
        "offramps": model.offramps.state_dict(),
    }
    torch.save(state, save_path)
    print(f"Saved joint weights to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/msmarco_train.parquet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--offramp_lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--max_steps", type=int, default=-1)
    args = parser.parse_args()

    train_joint(
        data_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        backbone_lr=args.backbone_lr,
        offramp_lr=args.offramp_lr,
        alpha=args.alpha,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )
