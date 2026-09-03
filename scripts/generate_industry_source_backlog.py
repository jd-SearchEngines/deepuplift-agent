from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.industry_playbook import industry_references


SOURCE_BACKLOG = [
    {
        "company": "Baidu",
        "search_target": "百度 / 广告 / 增益模型 / uplift / 因果",
        "status": "backlog",
        "reason": "current public search surfaced mostly generic or low-provenance material rather than a clear official industrial uplift deployment case.",
        "next_action": "only adopt if an official Baidu engineering/research post or peer-reviewed paper describes incrementality/uplift deployment with experiment design.",
    },
    {
        "company": "Xiaohongshu",
        "search_target": "小红书 / 增量 / 因果 / 推荐干预 / uplift",
        "status": "backlog",
        "reason": "no strong official uplift/incrementality landing article was found in the current pass.",
        "next_action": "keep as a sourcing target; prefer official tech blog, conference paper, or public engineering talk with clear treatment/control design.",
    },
    {
        "company": "Ctrip",
        "search_target": "携程 / 营销 / 用户增长 / uplift / 因果推断",
        "status": "backlog",
        "reason": "no high-confidence uplift deployment source was adopted in this pass.",
        "next_action": "search for official tech posts around coupon/CRM/marketing incrementality and keep only primary sources.",
    },
    {
        "company": "Meituan",
        "search_target": "美团 / 因果 / 智能营销 / uplift",
        "status": "adopted",
        "reason": "the ECUP paper is directly about full-chain intelligent marketing uplift and is stronger than generic causal overview articles.",
        "next_action": "keep ECUP as the primary citation; generic causal articles can be background but not core evidence.",
    },
    {
        "company": "Alibaba",
        "search_target": "阿里 / 阿里妈妈 / 淘宝 / 天猫 / 智能营销 / 增益模型",
        "status": "adopted",
        "reason": "Alibaba Cloud Developer Community material now covers AB-test heterogeneity, X-Learner/CATE/SHAP and EdgeRec request uplift; the older Alibaba Entertainment mirror remains only supplemental context.",
        "next_action": "keep searching for Alimama/Taobao/Tmall official ads or CRM incrementality posts, but the current Alibaba evidence is strong enough for core scenario mapping.",
    },
    {
        "company": "Kuaishou",
        "search_target": "快手 / uplift / short-video recommendation / multi-treatment",
        "status": "adopted",
        "reason": "CDUM and HMUM are research-quality recommendation uplift sources and map well to recommendation intervention and multi-treatment interview topics.",
        "next_action": "watch for official engineering writeups that confirm production deployment details.",
    },
    {
        "company": "ByteDance",
        "search_target": "字节 / uplift / delayed feedback / growth",
        "status": "adopted",
        "reason": "delayed-feedback uplift paper is directly useful for growth/push/activation delay and label-maturity questions.",
        "next_action": "keep searching for official engineering material around growth, ads, or recommendation incrementality.",
    },
]


def table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def build_markdown(generated_at: str) -> str:
    adopted = industry_references()
    adopted_rows = [
        {
            "company": ref["company"],
            "id": ref["id"],
            "source_type": ref["source_type"],
            "scenario": ref["scenario"],
            "url": ref["url"],
        }
        for ref in adopted
    ]
    return f"""# Uplift Industry Source Adoption And Backlog

Generated at: `{generated_at}`

This document records not only what the Agent adopted, but also what it deliberately did not adopt. The goal is to keep the knowledge base interview-defensible: official docs, company engineering/research posts, peer-reviewed papers, arXiv papers, and official open-source repos are preferred; weak reposts are treated as background only.

## Adopted Sources

{table(adopted_rows, ["company", "id", "source_type", "scenario", "url"])}

## Search Backlog And Adoption Decisions

{table(SOURCE_BACKLOG, ["company", "search_target", "status", "reason", "next_action"])}

## Interview Talking Point

If asked why the Agent does not include every article found online, say: "I separated source discovery from source adoption. The knowledge base only promotes sources with clear provenance and a treatment/control or causal decision angle; weaker articles remain in backlog until they can be verified or replaced."
"""


def main() -> None:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    output = Path("docs/UPLIFT_INDUSTRY_SOURCE_ADOPTION_BACKLOG.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(generated_at), encoding="utf-8")
    report = {
        "status": "ok",
        "generated_at": generated_at,
        "output": str(output),
        "adopted_sources": len(industry_references()),
        "backlog_rows": SOURCE_BACKLOG,
    }
    report_path = Path("reports/industry_source_backlog_latest.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
