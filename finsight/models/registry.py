from pathlib import Path
import json

def list_models(model_dir: Path):
    result = []
    for p in model_dir.rglob("metadata.json"):
        try:
            result.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            pass
    return result
