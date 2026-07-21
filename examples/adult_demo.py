from fairlens.datasets import load_adult_dataset
from fairlens.audit import FairnessAuditor

df = load_adult_dataset()

auditor = FairnessAuditor()

result = auditor.audit(
    dataframe=df,
    target_column="income",
    prediction_column="prediction",
    sensitive_feature="sex",
)

print(result)