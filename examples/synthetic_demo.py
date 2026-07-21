from fairlens.datasets import generate_synthetic_dataset
from fairlens.audit import FairnessAuditor

df = generate_synthetic_dataset()

auditor = FairnessAuditor()

result = auditor.audit(
    dataframe=df,
    target_column="Target",
    prediction_column="prediction",
    sensitive_feature="Gender",
)

print(result)