from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.model_cards import model_parameter_presets, model_source_and_family, recommended_model_preset
from deepuplift.core.registry import MODEL_REGISTRY


def main() -> None:
    failures = []
    source_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    for model in sorted(MODEL_REGISTRY):
        presets = model_parameter_presets(model)
        recommended_name, recommended_params = recommended_model_preset(model)
        source, family = model_source_and_family(model)
        source_counts[source] = source_counts.get(source, 0) + 1
        family_counts[family] = family_counts.get(family, 0) + 1
        if not presets:
            failures.append(f"{model}: no presets")
            continue
        if recommended_name not in presets:
            failures.append(f"{model}: recommended preset {recommended_name!r} not in presets")
        if recommended_params != presets.get(recommended_name, {}):
            failures.append(f"{model}: recommended params do not match preset payload")
        try:
            json.dumps(presets, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            failures.append(f"{model}: presets are not JSON serializable: {exc}")
    if failures:
        raise AssertionError("\n".join(failures))
    print(
        json.dumps(
            {
                "status": "ok",
                "models": len(MODEL_REGISTRY),
                "sources": source_counts,
                "families": family_counts,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
