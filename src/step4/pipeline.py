from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def run_step4(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "step4_summary.json"
    payload = {
        "step": 4,
        "name": "Three-layer validation",
        "status": "placeholder-complete",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "notes": "Implement FID, FreeSurfer biomarkers, and classifier utility validation.",
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return summary_path
