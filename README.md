# FairLens

**An open-source framework for auditing ML models for fairness.**

FairLens benchmarks [AIF360](https://github.com/Trusted-AI/AIF360) against
[Fairlearn](https://github.com/fairlearn/fairlearn) side by side on the same
model and dataset, maps the results onto **EU AI Act Annex III** high-risk
categories, and introduces the **Metric Agreement Index (MAI)** — a novel
measure of how much different fairness metrics and toolkits agree on
whether a model is "fair."

## Why

Two problems keep showing up in practical fairness auditing:

1. **Toolkits disagree.** AIF360 and Fairlearn implement overlapping
   metrics with different sign conventions and, in some cases, genuinely
   different definitions (e.g. Fairlearn's `equalized_odds_difference` is
   unsigned; AIF360's `average_odds_difference` is signed and averaged).
   A practitioner running just one library never sees this.
2. **Metrics disagree.** It's a well-established result in the fairness
   literature (Kleinberg et al. 2016, Chouldechova 2017) that different
   fairness definitions can be mutually incompatible — a model can pass
   demographic parity and fail equalized odds on the same data.

FairLens runs both libraries' metrics side by side and reports a single
**Metric Agreement Index** in `[0, 1]` that quantifies both kinds of
disagreement, so "the audit says the model is fair" becomes a more honest,
falsifiable claim.

## Install

```bash
pip install -e .
# or
pip install -r requirements.txt
```

Requires Python 3.10+. Core dependencies: `aif360`, `fairlearn`,
`scikit-learn`, `pandas`, `numpy`, `scipy`.

## Quickstart

```python
from fairlens.datasets import load_synthetic
from fairlens.audit import FairnessAudit
from fairlens.report import render_markdown

dataset = load_synthetic(n_samples=3000, bias_strength=0.4)

# Replace with your own model's predictions:
y_pred = your_model.predict(dataset.df.drop(columns=[dataset.label_name]))

audit = FairnessAudit()
result = audit.run(
    dataset,
    y_pred,
    use_case_description="credit scoring loan approval model",
)

print(render_markdown(result))
print("MAI:", result.mai_result.mai)
```

Or from the command line:

```bash
fairlens audit --dataset synthetic --bias-strength 0.5 \
    --use-case "resume screening tool for job applicants" \
    --out report.md
```

## Benchmark datasets

`fairlens.datasets` ships loaders for the three standard fairness
benchmarks:

| Dataset | Loader | Protected attribute | Task |
|---|---|---|---|
| UCI Adult Income | `load_adult()` | sex | income > $50K |
| ProPublica COMPAS | `load_compas()` | race | recidivism risk |
| UCI German Credit | `load_german()` | age | credit risk |

These wrap AIF360's dataset classes, which require the raw source files
(AIF360 doesn't bundle them for licensing reasons). The loaders will
attempt to download them automatically; if that fails in your network
environment, download manually per the URLs AIF360 prints, or call the
`fetch_*_raw_files()` helpers in `fairlens.datasets.loaders` directly.

`load_synthetic(n_samples, bias_strength)` generates a dataset with a
**known, tunable** amount of injected bias (`bias_strength` in `[0, 1]`),
useful for testing, demos, and validating that FairLens behaves as
expected on data with a known ground truth.

## The Metric Agreement Index (MAI)

MAI has two components:

- **Verdict agreement** — of every pairwise comparison across all computed
  metrics (both libraries, both metric families), what fraction agree on
  the binary fair/unfair call?
- **Cross-library severity-rank agreement** — for the metrics computed by
  *both* AIF360 and Fairlearn, does the relative severity ordering
  (Spearman correlation of normalized deviations) agree between the two
  implementations? (Requires ≥3 shared metrics; otherwise MAI falls back
  to verdict agreement alone.)

```
MAI = w_verdict * verdict_agreement + w_rank * rank_agreement   (defaults: 0.5 / 0.5)
```

The result is further decomposed into a **cross-library** component
(validates the toolkits agree with each other) and a **cross-metric**
component (surfaces genuine tension between fairness notions), since
conflating "the tools disagree" with "the definitions disagree" would
hide which problem is actually occurring. See
[`fairlens/mai/index.py`](fairlens/mai/index.py) for the full derivation.

## EU AI Act Annex III mapping

`fairlens.annex3.classify_annex3_risk(use_case_description, ...)` matches
a free-text use-case description against the eight Annex III high-risk
categories (biometrics, critical infrastructure, education, employment,
essential services, law enforcement, migration/border control,
justice/democratic processes) and flags the Article 10 / Article 15
obligations most relevant to a fairness audit. When passed the audit's
MAI result, it also flags cases where the audit is *not* one clean
fairness verdict but a genuinely contested one — directly relevant to
what belongs in the technical documentation required under the Act.

**This is a compliance-support aid, not legal advice.** It uses keyword
matching, is deliberately conservative (over- rather than under-flags),
and should be reviewed by qualified counsel before being relied on.

## Project status

Portfolio / research project. Metric Agreement Index write-up targeted
for an arXiv (cs.LG) submission and FAccT 2027 submission.

Author 
Priyal Parmar 
