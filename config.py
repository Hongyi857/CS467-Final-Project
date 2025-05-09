import os

HF_TOKEN = os.environ.get("HF_ACCESS_TOKEN", None)

# Data
DEV_SIZE = 7600
SEED = 42


# BERT fine‑tune
BERT_MODEL = "distilbert-base-uncased"
BERT_BATCH = 16
BERT_EPOCHS = 3
BERT_LR = 2e-5
BERT_MAX_LEN = 256

# In‑context
ICL_SHOTS = 1
