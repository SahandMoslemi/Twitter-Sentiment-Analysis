import os
import logging
from datetime import datetime
from tqdm import tqdm
import emoji


# Root
_ROOT_DIR = os.getcwd()

# Date Format
DATE = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Logs
_LOGS_DIR = os.path.join(_ROOT_DIR, "logs")
_LOG_DIR = os.path.join(_LOGS_DIR, f"{DATE}.log")

# Settings
logging.basicConfig(
    filename=_LOG_DIR, 
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
tqdm.pandas(desc="Progress ...")

# Parameters
NO_BPE_ITERATIONS = 2048
RANDOM_STATE = 1
EMBEDDING_DIM = 300
EMBEDDING_MODEL_EPOCHS = 40

# Data
EMOJIS = list(emoji.EMOJI_DATA.keys())

# Log
logging.info(f"no_bpe_iterations: {NO_BPE_ITERATIONS}")
logging.info(f"random_state: {RANDOM_STATE}")
logging.info(f"embedding_dim: {EMBEDDING_DIM}")
logging.info(f"embedding_model_epochs: {EMBEDDING_MODEL_EPOCHS}")
logging.info(f"emojis: {EMOJIS}")
