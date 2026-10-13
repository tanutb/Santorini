"""Export the served agent and its strength levels for in-browser play.

Writes web/models/<level>.bin (float32 tensors, little-endian, state_dict order) and
web/models/manifest.json (shapes, offsets, training metadata, recorded evaluations).
Run after promoting a checkpoint:  python scripts/export_models.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.rl.model import checkpoint_spec  # noqa: E402

CHECKPOINTS = ROOT / "checkpoints"
OUTPUT = ROOT / "web" / "models"
# Optional head-to-head win rates between levels, e.g. {"full": {"developing": 0.78}}: how often the
# level wins against another level, seat-balanced (server.rl.arena). Shown in the difficulty picker.
VERSUS = CHECKPOINTS / "headtohead.json"


def levels() -> tuple[str, list[dict]]:
    manifest = json.loads((CHECKPOINTS / "levels.json").read_text(encoding="utf8"))
    entries = []
    for entry in manifest["levels"]:
        path = CHECKPOINTS / ("ppo.pt" if entry["id"] == "full" else entry["checkpoint"])
        entries.append({"id": entry["id"], "name": entry["name"], "path": path})
    return manifest.get("default", "full"), entries


def export(entry: dict) -> dict:
    checkpoint = torch.load(entry["path"], map_location="cpu", weights_only=True)
    spec = checkpoint_spec(checkpoint)
    tensors, chunks, offset = [], [], 0
    for name, value in checkpoint["model"].items():
        array = value.detach().cpu().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(array.shape), "offset": offset})
        chunks.append(array.ravel())
        offset += array.size
    file = f"{entry['id']}.bin"
    np.concatenate(chunks).tofile(OUTPUT / file)
    metadata = checkpoint["metadata"]
    evaluation_path = entry["path"].with_suffix(".evaluation.json")
    evaluation = None
    if evaluation_path.exists():
        report = json.loads(evaluation_path.read_text(encoding="utf8"))
        if report.get("checkpoint_steps") == metadata.get("steps"):
            evaluation = {name: {key: result[key] for key in ("games", "wins", "losses", "draws", "win_rate", "confidence_95") if key in result}
                          for name, result in report["results"].items()}
    print(f"{file}: {spec}, {offset:,} parameters, {(OUTPUT / file).stat().st_size / 1024:.0f} KiB")
    versus = json.loads(VERSUS.read_text(encoding="utf8")).get(entry["id"], {}) if VERSUS.exists() else {}
    return {"id": entry["id"], "name": entry["name"], "file": file, "spec": spec, "parameters": offset, "tensors": tensors,
            "sha256": hashlib.sha256((OUTPUT / file).read_bytes()).hexdigest(),
            "metadata": {key: metadata.get(key) for key in ("steps", "episodes", "updates", "algorithm")},
            "evaluation": evaluation, "versus": versus}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    default, entries = levels()
    manifest = {"format": 1, "default": default, "levels": [export(entry) for entry in entries]}
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
