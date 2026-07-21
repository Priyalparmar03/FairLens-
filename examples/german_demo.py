from fairlens.datasets import load_german_dataset
from fairlens.audit import FairnessAuditor

df = load_german_dataset()

auditor = FairnessAuditor()

result = auditor.audit(
    dataframe=df,
    target_column="Target",
    prediction_column="prediction",
    sensitive_feature="Age",
)

print(result)