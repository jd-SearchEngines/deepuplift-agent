from __future__ import annotations

import json
from pathlib import Path


def llm_routing_policy(
    *,
    requests: int,
    quality_uplift: float,
    value_per_quality: float,
    route_fraction: float,
    small_cost_per_1k: float,
    strong_cost_per_1k: float,
    latency_penalty: float,
    router_precision: float,
) -> dict[str, float | int]:
    routed_requests = int(float(requests) * float(route_fraction))
    incremental_cost = routed_requests * ((strong_cost_per_1k - small_cost_per_1k) / 1000.0 + latency_penalty)
    incremental_value = routed_requests * quality_uplift * value_per_quality * router_precision
    all_strong_cost = requests * ((strong_cost_per_1k - small_cost_per_1k) / 1000.0 + latency_penalty)
    all_strong_value = requests * quality_uplift * value_per_quality
    return {
        "requests": requests,
        "routed_requests": routed_requests,
        "incremental_value": incremental_value,
        "incremental_cost": incremental_cost,
        "net_value": incremental_value - incremental_cost,
        "all_strong_net_value": all_strong_value - all_strong_cost,
        "route_fraction": route_fraction,
    }


def main() -> None:
    base = llm_routing_policy(
        requests=10_000,
        quality_uplift=0.08,
        value_per_quality=2.0,
        route_fraction=0.20,
        small_cost_per_1k=0.20,
        strong_cost_per_1k=4.00,
        latency_penalty=0.002,
        router_precision=0.75,
    )
    high_precision = llm_routing_policy(
        requests=10_000,
        quality_uplift=0.08,
        value_per_quality=2.0,
        route_fraction=0.20,
        small_cost_per_1k=0.20,
        strong_cost_per_1k=4.00,
        latency_penalty=0.002,
        router_precision=0.95,
    )
    low_value = llm_routing_policy(
        requests=10_000,
        quality_uplift=0.02,
        value_per_quality=1.0,
        route_fraction=0.20,
        small_cost_per_1k=0.20,
        strong_cost_per_1k=4.00,
        latency_penalty=0.002,
        router_precision=0.75,
    )
    failures: list[str] = []
    if base["routed_requests"] != 2000:
        failures.append("routed request count should follow route_fraction")
    if high_precision["net_value"] <= base["net_value"]:
        failures.append("higher router precision should improve net value")
    if low_value["net_value"] >= base["net_value"]:
        failures.append("lower quality/value assumptions should reduce net value")
    payload = {
        "status": "fail" if failures else "ok",
        "scenarios": {
            "base": base,
            "high_precision": high_precision,
            "low_value": low_value,
        },
        "failures": failures,
    }
    output = Path("reports/llm_routing_policy_smoke_latest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
