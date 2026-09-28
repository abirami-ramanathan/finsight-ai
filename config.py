from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DOCUMENTS_DIR = DATA_DIR / "documents"
MODEL_DIR = ROOT / os.getenv("MODEL_DIR", "models")
OUTPUT_DIR = ROOT / os.getenv("OUTPUT_DIR", "outputs")

for p in [RAW_DIR, PROCESSED_DIR, DOCUMENTS_DIR, MODEL_DIR, OUTPUT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

LOOKBACK = int(os.getenv("LOOKBACK", "30"))
EPOCHS = int(os.getenv("EPOCHS", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))

PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.20"))
KL_THRESHOLD = float(os.getenv("KL_THRESHOLD", "0.20"))
ERROR_WINDOW = int(os.getenv("ERROR_WINDOW", "30"))

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
BANDIT_ALPHA = float(os.getenv("BANDIT_ALPHA", "1.0"))
