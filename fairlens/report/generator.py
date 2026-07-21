"""Renders an AuditResult as a human-readable Markdown report."""
from __future__ import annotations

from fairlens.audit import AuditResult
from fairlens.metrics.common import REGISTRY


def render_markdown(result: AuditResult) -> str:
    lines = [f"# FairLens Audit Report -- {result.dataset_name}", ""]

    lines.append("## Metric comparison: AIF360 vs Fairlearn\n")
    lines.append("| Metric | AIF360 | Fairlearn | Fair? (AIF360) | Fair? (Fairlearn) |")
    lines.append("|---|---|---|---|---|")

    all_keys = sorted(set(result.aif360_metrics) | set(result.fairlearn_metrics))
    for key in all_keys:
        spec = REGISTRY[key]
        a = result.aif360_metrics.get(key)
        f = result.fairlearn_metrics.get(key)
        a_str = f"{a:.4f}" if a is not None else "n/a"
        f_str = f"{f:.4f}" if f is not None else "n/a"
        a_fair_str = ("yes" if spec.is_fair(a) else "**NO**") if a is not None else "n/a"
        f_fair_str = ("yes" if spec.is_fair(f) else "**NO**") if f is not None else "n/a"
        lines.append(f"| {spec.display_name} | {a_str} | {f_str} | {a_fair_str} | {f_fair_str} |")

    lines.append("")
    lines.append("## Metric Agreement Index (MAI)\n")
    m = result.mai_result
    lines.append(f"- **Overall MAI: {m.mai:.3f}** (1.0 = full agreement, 0.0 = maximal disagreement)")
    lines.append(f"- Verdict agreement (fair/unfair concordance): {m.mai_verdict:.3f}")
    lines.append(
        f"- Severity-rank agreement: {m.mai_rank:.3f}" if m.mai_rank is not None
        else "- Severity-rank agreement: n/a (fewer than 3 evaluations)"
    )
    if m.cross_library_mai is not None:
        lines.append(f"- Cross-library agreement (AIF360 vs Fairlearn, same metric): {m.cross_library_mai:.3f}")
    if m.cross_metric_mai is not None:
        lines.append(f"- Cross-metric agreement (different fairness notions): {m.cross_metric_mai:.3f}")

    if result.unfair_metric_keys:
        lines.append("")
        lines.append(f"**Metrics outside their fairness threshold:** {', '.join(result.unfair_metric_keys)}")

    if result.annex3:
        a3 = result.annex3
        lines.append("")
        lines.append("## EU AI Act Annex III Mapping\n")
        lines.append(f"- **Risk tier:** {a3.risk_tier}")
        if a3.matched_categories:
            cats = ", ".join(f"#{cid} {name}" for cid, name in a3.matched_categories)
            lines.append(f"- **Matched categories:** {cats}")
        if a3.relevant_articles:
            lines.append(f"- **Relevant articles:** {', '.join(a3.relevant_articles)}")
        lines.append(f"- {a3.rationale}")
        if a3.fairness_flag:
            lines.append(f"- \u26a0\ufe0f {a3.fairness_flag}")

    lines.append("")
    lines.append(
        "_This report is a technical fairness audit and is not legal advice. "
        "EU AI Act classification should be confirmed with qualified counsel._"
    )
    return "\n".join(lines)
