from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_DIR
from finsight.models.registry import list_models
for x in list_models(MODEL_DIR):
    print(x)
