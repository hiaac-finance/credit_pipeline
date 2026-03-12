# Credit Pipeline

`credit_pipeline` is a Python package for building and analyzing credit-scoring workflows with an emphasis on trustworthy machine learning. The codebase combines:

- tabular preprocessing and model training,
- hyperparameter optimization,
- performance and fairness evaluation,
- fairness-aware learning strategies,
- reject inference methods, and
- model explainability and counterfactual analysis.

The repository accompanies research on responsible machine learning for credit scoring and provides reusable code for experiments on German Credit, Taiwan, and Home Credit datasets. To see the original version that accompanied the paper, see the branch `ncaa_experiments`.

## What is included

The package lives in [src/credit_pipeline](src/credit_pipeline) and is organized into the following modules:

| Module | Purpose | Main objects |
| --- | --- | --- |
| [src/credit_pipeline/data.py](src/credit_pipeline/data.py) | Dataset download, preparation, and loading | `download_datasets()`, `prepare_datasets()`, `load_dataset()` |
| [src/credit_pipeline/training.py](src/credit_pipeline/training.py) | Preprocessing pipelines and Optuna-based tuning | `EBE`, `create_pipeline()`, `optimize_model()`, `optimize_model_fast()`, `ks_threshold()` |
| [src/credit_pipeline/models.py](src/credit_pipeline/models.py) | Extra models with a scikit-learn style API | `MLPClassifier` |
| [src/credit_pipeline/evaluate.py](src/credit_pipeline/evaluate.py) | Performance, threshold, and fairness metrics | `get_metrics()`, `get_fairness_metrics()`, fairness scorers |
| [src/credit_pipeline/fairness_models.py](src/credit_pipeline/fairness_models.py) | Fairness-aware learning strategies | `Reweighing`, `FairGBM`, `ThresholdOpt` |
| [src/credit_pipeline/reject_inference.py](src/credit_pipeline/reject_inference.py) | Reject inference algorithms for accepted/rejected populations | `RejectUpward`, `RejectDownward`, `RejectSoftCutoff`, `FuzzyParcelling`, `RejectExtrapolation`, `RejectSpreading` |
| [src/credit_pipeline/explainability.py](src/credit_pipeline/explainability.py) | Global/local explanations and counterfactuals | `PartialDependencePipeline`, `ShapPipelineExplainer`, `LimePipelineExplainer`, `MAPOCAM`, `Dice`, `display_cfs()` |

## Installation

### Requirements

- Python 3.9+
- A local clone of this repository

### Install the package

From the repository root:

```bash
pip install .
```

Core dependencies are declared in [pyproject.toml](pyproject.toml).

## Data

The repository already contains prepared CSV files in [data/prepared](data/prepared) for the supported benchmark datasets:

- German Credit
- Taiwan Credit Card Default
- Home Credit

Raw data files are also included under [data](data). The helper functions in [src/credit_pipeline/data.py](src/credit_pipeline/data.py) can be used to download or preprocess datasets when needed.

## Quick start

The example below shows the typical workflow: load a dataset, create a preprocessing/training pipeline, fit a classifier, and compute metrics.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_pipeline.training import create_pipeline, ks_threshold
from credit_pipeline.evaluate import get_metrics

# Example with a prepared dataset stored in the repository
df = pd.read_csv("data/prepared/german.csv")

X = df.drop(columns=["DEFAULT"])
y = df["DEFAULT"]

pipeline = create_pipeline(
   X,
   y,
   classifier=LogisticRegression(max_iter=1000, solver="liblinear"),
)
pipeline.fit(X, y)

threshold = ks_threshold(y, pipeline.predict_proba(X)[:, 1])
metrics = get_metrics({"logistic_regression": [pipeline, threshold]}, X, y)
print(metrics)
```

## Main capabilities

### 1. Training pipelines

The training utilities build end-to-end scikit-learn pipelines for heterogeneous credit data.

Features include:

- missing-value imputation for numeric and categorical columns,
- ordinal encoding and optional one-hot encoding,
- numeric standardization,
- EBE target encoding for higher-cardinality categorical variables, and
- direct integration with scikit-learn compatible classifiers.

Use `create_pipeline()` when you want a ready-to-fit preprocessing + model pipeline.

### 2. Hyperparameter optimization

The package provides two Optuna-based tuning functions:

- `optimize_model()` for full pipeline optimization,
- `optimize_model_fast()` for faster tuning on preprocessed validation data.

Default search spaces are included for:

- `LogisticRegression`
- `RandomForestClassifier`
- `LGBMClassifier`
- `MLPClassifier`

### 3. Evaluation and fairness metrics

The evaluation module contains standard classification metrics and fairness metrics commonly used in credit modeling. The helpers `create_eod_scorer()` and `create_fairness_scorer()` can be used during model selection.

### 4. Fairness-aware modeling

The fairness module includes strategies across the pipeline:

- **Pre-processing**: `Reweighing`
- **In-processing**: `FairGBM`
- **Post-processing**: `ThresholdOpt`

These classes follow a scikit-learn-like interface and are designed for binary classification with sensitive attributes supplied during training or prediction.

### 5. Reject inference

The reject inference module implements several techniques for scenarios where only accepted applicants have observed repayment labels and rejected applicants remain unlabeled.

Available methods include: `RejectUpward`, `RejectDownward`,  `RejectSoftCutoff`, `FuzzyParcelling`, `RejectExtrapolation`, `RejectSpreading`.

These classes expect a combined feature matrix where unlabeled observations are marked with `-1` in the target vector.

### 6. Explainability and counterfactuals

The explainability module supports both global and local interpretation:

- partial dependence and ICE curves via `PartialDependencePipeline`,
- SHAP explanations via `ShapPipelineExplainer`,
- LIME explanations via `LimePipelineExplainer`,
- actionable counterfactual search with `MAPOCAM`, and
- counterfactual generation with `Dice`.

`display_cfs()` formats generated counterfactuals into a comparison table.

## Example notebooks

The notebooks in [examples](examples) show how the package is intended to be used:

- [examples/0_training_model.ipynb](examples/0_training_model.ipynb) — model training
- [examples/1_fairness_methods.ipynb](examples/1_fairness_methods.ipynb) — fairness methods
- [examples/2_reject_inference.ipynb](examples/2_reject_inference.ipynb) — reject inference
- [examples/3_explainability.ipynb](examples/3_explainability.ipynb) — explainability

These notebooks are the best entry point for understanding expected inputs, preprocessing choices, and end-to-end workflows.

## Citation and context

If using this code, please cite the associated research paper:

```
@article{valdrighi2025best,
  title={Best practices for responsible machine learning in credit scoring},
  author={Valdrighi, Giovani and M. Ribeiro, Athyrson and SB Pereira, Jansen and Guardieiro, Vitoria and Hendricks, Arthur and Miranda Filho, D{\'e}cio and Nieto Garcia, Juan David and F. Bocca, Felipe and B. Veronese, Thalita and Wanner, Lucas and others},
  journal={Neural Computing and Applications},
  volume={37},
  number={25},
  pages={20781--20821},
  year={2025},
  publisher={Springer}
}
```

## Contact

Project coordinator: Marcos Medeiros Raimundo  
Email: mraimundo@ic.unicamp.br
