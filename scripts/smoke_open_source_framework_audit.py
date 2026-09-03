from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.open_source_frameworks import (
    open_source_framework_counts,
    open_source_framework_rows,
    open_source_framework_agent_reply,
)


REQUIRED = {
    "EconML",
    "CausalML",
    "scikit-uplift",
    "UTBoost",
    "pylift",
    "grf",
    "DoWhy / PyWhy",
    "DoubleML",
    "YLearn",
    "causallib",
    "CausalTune",
    "GeoLift",
    "UpliftML",
    "CausalLift",
    "causal-learn",
    "RouteLLM",
    "Open Bandit Pipeline",
    "Vowpal Wabbit",
    "LangGraph Open Deep Research",
    "GPT Researcher",
    "Local Deep Researcher",
}


def main() -> None:
    rows = open_source_framework_rows()
    names = {row["framework"] for row in rows}
    missing = sorted(REQUIRED - names)
    if missing:
        raise SystemExit(f"missing framework audit rows: {missing}")
    counts = open_source_framework_counts()
    if counts["total"] < 24 or counts["ready"] < 3 or counts["guarded"] < 2 or counts["optional"] < 9:
        raise SystemExit(f"framework audit counts too small: {counts}")
    for field in ["supported_models", "treatment_support", "scenario_fit", "local_status", "integration_path", "limitations", "url", "github_repo", "github_activity"]:
        bad = [row["framework"] for row in rows if not str(row.get(field, "")).strip()]
        if bad:
            raise SystemExit(f"blank {field}: {bad}")

    reply = open_source_framework_agent_reply("当前用了哪些 GitHub 开源框架，RouteLLM Open Deep Research OBP 哪些 ready guarded optional，有什么限制？")
    if not reply or "EconML" not in reply or "RouteLLM" not in reply or "guarded" not in reply or "Evidence" not in reply:
        raise SystemExit(f"framework agent reply incomplete: {reply}")

    out = Path("reports/open_source_framework_audit_smoke_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "pass", "counts": counts, "frameworks": sorted(names)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
