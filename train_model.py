
import json
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

BASE = Path(__file__).parent
DATA = BASE / "data"
MODELS = BASE / "models"
RESULTS = BASE / "results"

for folder in (DATA, MODELS, RESULTS):
    folder.mkdir(parents=True, exist_ok=True)

random.seed(42)

phishing_templates = [
    "Your account needs verification before access continues",
    "A payment is waiting for confirmation",
    "Your mailbox storage requires attention",
    "A security alert requires you to review your account",
    "Your parcel delivery needs additional details",
    "An invoice is waiting for payment approval",
    "Your account session needs to be renewed",
    "A reward claim requires confirmation of eligibility",
    "Your subscription payment could not be completed",
    "A document is waiting for your approval",
    "Your account access will be limited pending review",
    "A transaction requires confirmation before processing",
]

legitimate_templates = [
    "The team meeting has been moved to tomorrow",
    "Your library book is due next week",
    "The class timetable is available on the college portal",
    "Your appointment reminder is ready to view",
    "The monthly report is available for review",
    "Your order receipt is available in your account",
    "The project discussion starts at the scheduled time",
    "Your subscription renewal receipt is available",
    "The office will close early on Friday",
    "Your exam timetable has been published",
    "The team has shared the meeting notes",
    "Your delivery update is available in the order history",
]

openings = [
    "",
    "Hello, ",
    "Notice: ",
    "Update: ",
    "Please note: ",
]

phishing_additions = [
    " Please review the request using [URL_REDACTED].",
    " The sender requests confirmation of account information.",
    " Review the request soon before access is affected.",
    " Do not share passwords or verification codes in a reply.",
]

legitimate_additions = [
    " Please check the usual official portal for details.",
    " Contact the relevant office if you have questions.",
    " No reply is required unless you need assistance.",
    " You can review the information during normal hours.",
]


def make_examples(templates, additions, label, prefix):
    rows = []

    for campaign_number, template in enumerate(templates, start=1):
        campaign = f"{prefix}_campaign_{campaign_number:02d}"

        # 5 openings x 4 additions = 20 unique message variations
        for opening_index, opening in enumerate(openings):
            for addition_index, addition in enumerate(additions):
                body = opening + template + addition

                rows.append({
                    "message_id": (
                        f"{campaign}_"
                        f"{opening_index * len(additions) + addition_index + 1:03d}"
                    ),
                    "channel": (
                        "email"
                        if (opening_index + addition_index) % 2 == 0
                        else "sms"
                    ),
                    "subject_or_title": template,
                    "body_text": body,
                    "sanitized_url_features": (
                        "redacted_url_present"
                        if "[URL_REDACTED]" in body
                        else "no_url_detected"
                    ),
                    "sender_domain_type": (
                        "unknown_synthetic"
                        if label == 1
                        else "ordinary_synthetic"
                    ),
                    "label": label,
                    "campaign_group": campaign,
                    "collected_date": "2026-10-26",
                    "provenance": "synthetic_generated_example",
                })

    return rows


rows = (
    make_examples(phishing_templates, phishing_additions, 1, "phish")
    + make_examples(legitimate_templates, legitimate_additions, 0, "legit")
)

df = pd.DataFrame(rows)
before_dedup = len(df)

df = df.drop_duplicates(subset=["body_text"]).reset_index(drop=True)

df.to_csv(DATA / "messages.csv", index=False)

manifest = pd.DataFrame([{
    "dataset": "PhishShift synthetic prototype dataset",
    "record_count": len(df),
    "records_before_deduplication": before_dedup,
    "source_type": "locally generated synthetic examples",
    "external_dataset_used": "No",
    "live_urls_used": "No; URLs are redacted placeholders",
    "label_meaning": "1 = suspicious scenario; 0 = legitimate scenario",
    "limitation": (
        "Synthetic examples do not represent real-world performance "
        "or prove reliable phishing detection."
    ),
}])

manifest.to_csv(DATA / "source_manifest.csv", index=False)

# Keep entire campaigns separate between training and testing.
df["campaign_number"] = (
    df["campaign_group"].str.extract(r"(\d+)$").astype(int)
)

test_mask = df["campaign_number"] >= 10
train_df = df[~test_mask].copy()
test_df = df[test_mask].copy()

X_train = train_df["body_text"]
y_train = train_df["label"]
X_test = test_df["body_text"]
y_test = test_df["label"]


def rule_predict(message):
    text = message.lower()
    cues = [
        "verify", "payment", "security alert", "confirm",
        "limited", "claim", "invoice", "[url_redacted]",
        "renewed", "transaction",
    ]
    score = sum(cue in text for cue in cues)
    return 1 if score >= 2 else 0


models = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Multinomial Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", MultinomialNB()),
    ]),
}

predictions = {
    "Rule Baseline": X_test.apply(rule_predict).to_numpy()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions[name] = model.predict(X_test)

results = []

for name, pred in predictions.items():
    tn, fp, fn, tp = confusion_matrix(
        y_test, pred, labels=[0, 1]
    ).ravel()

    results.append({
        "model": name,
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, pred, zero_division=0), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    })

comparison = pd.DataFrame(results)
comparison.to_csv(RESULTS / "model_comparison.csv", index=False)

test_output = test_df[
    ["message_id", "body_text", "label", "campaign_group"]
].copy()

for name, pred in predictions.items():
    safe_name = name.lower().replace(" ", "_")
    test_output[safe_name + "_prediction"] = pred

test_output.to_csv(RESULTS / "test_predictions.csv", index=False)

joblib.dump(models["Logistic Regression"], MODELS / "best_model.joblib")

with open(RESULTS / "metrics.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=4)

print("\nPhishShift training completed!")
print(f"Records before deduplication: {before_dedup}")
print(f"Dataset records after deduplication: {len(df)}")
print(f"Training records: {len(train_df)}")
print(f"Test records: {len(test_df)}")
print(f"Unique message bodies: {df['body_text'].nunique()}")
print(
    "Campaign overlap between train and test:",
    bool(set(train_df["campaign_group"]) & set(test_df["campaign_group"]))
)
print("\nModel comparison:")
print(comparison.to_string(index=False))
print("\nFiles saved in data/, models/, and results/.")
print("Reminder: all records are synthetic; results are not real-world validation.")
