from fairlens.datasets import load_compas_dataset
from fairlens.audit import FairnessAuditor

df = load_compas_dataset()

auditor = FairnessAuditor()

result = auditor.audit(
    dataframe=df,
    target_column="two_year_recid",
    prediction_column="prediction",
    sensitive_feature="race",
)

print(result)