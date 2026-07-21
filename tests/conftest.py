import pytest
import pandas as pd


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        "target": [1, 0, 1, 0, 1, 0],
        "prediction": [1, 0, 1, 1, 0, 0],
        "sex": ["M", "F", "M", "F", "M", "F"],
    })


@pytest.fixture
def dataset_info():
    return {
        "name": "Synthetic",
        "samples": 6,
        "protected_attribute": "sex",
    }