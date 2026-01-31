# Enhanced Benchmark Tool

A small Python package for **dataset profiling** and **model benchmarking** (scikit-learn compatible).

## Install (dev)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Quickstart

### Dataset profiling

```python
from enhanced_benchmark_tool import profile_dataset

profile = profile_dataset("data.csv")
print(profile["shape"], profile["columns"])
```

### Model benchmarking

```python
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from enhanced_benchmark_tool import benchmark_model

df = pd.DataFrame({"A": [1,2,3,4], "B": [5,6,7,8], "target": [0,1,0,1]})
X = df[["A","B"]]
y = df["target"]

metrics = benchmark_model(DecisionTreeClassifier(), X, y)
print(metrics["accuracy"], metrics["training_time"])
```

## CLI

```bash
ebt-profile --input data.csv
```
