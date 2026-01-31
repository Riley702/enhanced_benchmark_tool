from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def profile_dataset(file_path: str) -> Dict[str, Any]:
    """Generate a basic statistical profile for a CSV dataset.

    Args:
        file_path: Path to a CSV file.

    Returns:
        A dictionary containing shape, columns, null counts, dtypes, and summary stats.
    """
    data = pd.read_csv(file_path)
    return {
        "shape": data.shape,
        "columns": list(data.columns),
        "null_counts": data.isnull().sum().to_dict(),
        "data_types": {k: str(v) for k, v in data.dtypes.to_dict().items()},
        "summary": data.describe(include="all").to_dict(),
    }
