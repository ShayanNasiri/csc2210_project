"""Project-wide constants."""

# Model configuration
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NUM_BERT_LAYERS = 6
NUM_OFFRAMPS = 5
HIDDEN_SIZE = 384
MAX_TOKEN_LENGTH = 512

# Inference configuration
WARMUP_BATCHES = 10
TIMED_BATCH_LIMIT = 100
DEFAULT_BATCH_SIZE = 64

# Default paths
DEFAULT_DEV_DATA_PATH = "data/dev_tokenized.pt"
DEFAULT_VAL_DATA_PATH = "data/val_tokenized.pt"
DEFAULT_TRAIN_SPLIT_PATH = "data/msmarco_train_split.parquet"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_OFFRAMP_WEIGHTS_PATH = "results/offramp_weights.pt"
DEFAULT_JOINT_WEIGHTS_PATH = "results/joint_weights.pt"
DEFAULT_SYSTEM_E_WEIGHTS_PATH = "results/system_e_joint_distill_weights.pt"

# Default thresholds for Baseline B
DEFAULT_ENTROPY_THRESHOLDS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]

# Per-ramp threshold grid for System F sweep (log-spaced 7 values, ~3x steps)
DEFAULT_PER_RAMP_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]
