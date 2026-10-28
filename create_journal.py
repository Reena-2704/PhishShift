
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

experiments = [
    {
        "experiment_id": "EXP-01",
        "experiment": "Synthetic dataset generation",
        "purpose": "Create labelled examples with campaign groups",
        "outcome": "480 unique messages; 240 per label",
        "limitation": "Synthetic examples only",
    },
    {
        "experiment_id": "EXP-02",
        "experiment": "Campaign-separated model comparison",
        "purpose": "Compare rules, Logistic Regression and Naive Bayes",
        "outcome": "Rule baseline 75%; both ML models 100% on this test set",
        "limitation": "Repeated synthetic wording may inflate scores",
    },
    {
        "experiment_id": "EXP-03",
        "experiment": "Acceptance and robustness tests",
        "purpose": "Probe unfamiliar wording, URL patterns and legitimate notices",
        "outcome": "Five manually written cases tested; scores saved",
        "limitation": "Small, manually selected test set",
    },
    {
        "experiment_id": "EXP-04",
        "experiment": "EDA and data-quality checks",
        "purpose": "Inspect labels, channels, campaigns and data quality",
        "outcome": "480 unique bodies and zero missing values",
        "limitation": "Dataset balance is synthetic, not representative",
    },
    {
        "experiment_id": "EXP-05",
        "experiment": "Error analysis",
        "purpose": "Inspect mistakes on held-out campaigns",
        "outcome": "30 rule-baseline errors; zero ML-model errors",
        "limitation": "Zero observed errors does not prove general reliability",
    },
]

journal = pd.DataFrame(experiments)
path = RESULTS / "experiment_journal.csv"
journal.to_csv(path, index=False)

print("Experiment journal created:")
print(path)
print("\nExperiments recorded:", len(journal))
print(journal[["experiment_id", "experiment"]].to_string(index=False))
