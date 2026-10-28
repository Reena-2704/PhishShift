
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "best_model.joblib"
DATA_PATH = BASE / "data" / "messages.csv"
RESULTS_PATH = BASE / "results"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Run python train_model.py first.")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# These are manually written acceptance examples.
# They are NOT part of the model's training data.
tests = [
    {
        "test_name": "Legitimate urgent password reset",
        "message": (
            "Urgent: The IT team has scheduled a password reset. "
            "Visit your usual company portal or contact IT support."
        ),
        "expected": "legitimate_or_uncertain",
    },
    {
        "test_name": "Unfamiliar suspicious wording",
        "message": (
            "Your digital access token is awaiting reconciliation. "
            "Use [URL_REDACTED] to review the pending request."
        ),
        "expected": "suspicious_or_uncertain",
    },
    {
        "test_name": "Defanged URL pattern",
        "message": (
            "Your account requires review at hxxps[:]//example[.]invalid"
        ),
        "expected": "suspicious_or_uncertain",
    },
    {
        "test_name": "Punctuation and spacing variation",
        "message": (
            "A c c o u n t  r e v i e w !!! "
            "Your access may change; see [URL_REDACTED]"
        ),
        "expected": "suspicious_or_uncertain",
    },
    {
        "test_name": "Routine legitimate notice",
        "message": (
            "The college timetable is available on the usual student portal."
        ),
        "expected": "legitimate_or_uncertain",
    },
]

rows = []

for item in tests:
    message = item["message"]
    score = float(model.predict_proba([message])[0][1])

    if score >= 0.65:
        result = "POTENTIALLY SUSPICIOUS"
    elif score <= 0.35:
        result = "NO STRONG SUSPICIOUS PATTERN"
    else:
        result = "UNCERTAIN"

    rows.append({
        "test_name": item["test_name"],
        "message": message,
        "expected_behavior": item["expected"],
        "suspicious_score": round(score, 4),
        "scanner_result": result,
        "within_uncertain_band": 0.35 < score < 0.65,
    })

results = pd.DataFrame(rows)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)
output_path = RESULTS_PATH / "acceptance_test_results.csv"
results.to_csv(output_path, index=False)

print("\nPHISSHIFT ACCEPTANCE TEST RESULTS")
print(results[
    ["test_name", "suspicious_score", "scanner_result"]
].to_string(index=False))
print(f"\nResults saved to: {output_path}")
print("\nReview each result manually; these tests do not prove real-world accuracy.")
