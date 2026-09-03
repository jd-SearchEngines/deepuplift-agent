from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.model_deconstruction import write_model_deconstruction_docs  # noqa: E402


def main() -> None:
    payload = write_model_deconstruction_docs(ROOT)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
