"""Default hyperparameters for Step 0 (see step0.md)."""

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

PART_A_NUM_PROBLEMS = 30
N_BRANCHES = 16
CHUNK_SIZE = 50
PART_A_MAX_NEW_TOKENS = 8192

PART_C_NUM_PROBLEMS = 25
PART_C_MAX_NEW_TOKENS = 500
PART_C_CHECKPOINTS = list(range(50, 501, 50))

STEP1_NUM_PROBLEMS = 50
STEP1_MAX_NEW_TOKENS = 8192
STEP1_CHECKPOINTS = [
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    750,
    1000,
    1500,
    2000,
    3000,
    4000,
    5000,
    6000,
    7000,
    8000,
]

LEVERAGE_PROJ_DIM = 128
LEVERAGE_SVD_RANK = 8

TEMPERATURE = 0.6
SPEARMAN_THRESHOLD = 0.80

DATA_REL_PATH = "data/math_500/test.jsonl"
RESULTS_STEP0_DIR = "results/step_0_analysis"

# Appended to every MATH user message (Hugging Face model card recommendation).
MATH_USER_PROMPT_SUFFIX = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Attention: prefer Flash Attention 2 on CUDA when installed (see src/model_loader.py).
# Override with env ATTN_IMPLEMENTATION=sdpa|eager|flash_attention_2
ATTN_IMPLEMENTATION = "flash_attention_2"
