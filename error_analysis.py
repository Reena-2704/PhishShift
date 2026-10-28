
from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
RESULTS = BASE / "results"

predictions_path = RESULTS / "test_predictions.csv"
output_path = RESULTS / "error_analysis.csv"

df = pd.read_csv(predictions_path)

model_columns = [
    "rule_baseline_prediction",
    "logistic_regression_prediction",
    "multinomial_naive_bayes_prediction",
]

for column in model_columns:
    if column not in df.columns:
        raise ValueError(f"Missing prediction column: {column}")

# Record errors made by each model.
error_rows = []

for _, row in df.iterrows():
    for column in model_columns:
        actual = int(row["label"])
        predicted = int(row[column])

        if actual != predicted:
            error_rows.append({
                "model": column.replace("_prediction", ""),
                "message_id": row["message_id"],
                "campaign_group": row["campaign_group"],
                "actual_label": actual,
                "predicted_label": predicted,
                "error_type": (
                    "False Positive" if predicted == 1
                    else "False Negative"
                ),
                "message": row["body_text"],
            })

errors = pd.DataFrame(
    error_rows,
    columns=[
        "model", "message_id", "campaign_group",
        "actual_label", "predicted_label",
        "error_type", "message"
    ]
)

errors.to_csv(output_path, index=False)

print("MODEL ERROR ANALYSIS")
print("--------------------")

for column in model_columns:
    model_name = column.replace("_prediction", "")
    count = int((df[column] != df["label"]).sum())
    print(f"{model_name}: {count} errors out of {len(df)} test messages")

if errors.empty:
    print("\nNo errors were found on this synthetic test set.")
    print("This does not establish real-world reliability.")
else:
    print(f"\nTotal error records across models: {len(errors)}")
    print("\nError examples:")
    print(errors.head(10).to_string(index=False))

print(f"\nSaved error analysis to: {output_path}")
