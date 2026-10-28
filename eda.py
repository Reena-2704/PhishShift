
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "messages.csv"
RESULTS_PATH = BASE / "results"
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Check the dataset before plotting.
required_columns = [
    "message_id", "channel", "body_text", "label",
    "campaign_group", "provenance"
]

missing_columns = [
    column for column in required_columns if column not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("DATASET OVERVIEW")
print("----------------")
print(f"Total messages: {len(df)}")
print(f"Unique message bodies: {df['body_text'].nunique()}")
print(f"Duplicate message bodies: {df['body_text'].duplicated().sum()}")
print(f"Missing values: {df.isna().sum().sum()}")
print("\nLabel counts:")
print(df["label"].map({
    0: "Legitimate",
    1: "Suspicious"
}).value_counts())

# Visualization 1: Message label distribution
label_counts = df["label"].map({
    0: "Legitimate",
    1: "Suspicious"
}).value_counts()

plt.figure(figsize=(7, 5))
label_counts.plot(kind="bar")
plt.title("Figure 1: Distribution of Message Labels")
plt.xlabel("Message label")
plt.ylabel("Number of messages")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_PATH / "eda_label_distribution.png", dpi=150)
plt.close()

# Visualization 2: Channel distribution by label
channel_counts = pd.crosstab(
    df["channel"],
    df["label"].map({0: "Legitimate", 1: "Suspicious"})
)

channel_counts.plot(kind="bar", figsize=(7, 5))
plt.title("Figure 2: Message Channel by Label")
plt.xlabel("Message channel")
plt.ylabel("Number of messages")
plt.xticks(rotation=0)
plt.legend(title="Label")
plt.tight_layout()
plt.savefig(RESULTS_PATH / "eda_channel_by_label.png", dpi=150)
plt.close()

# Visualization 3: Campaign sizes by label
campaign_counts = (
    df.groupby(["campaign_group", "label"])
      .size()
      .unstack(fill_value=0)
      .rename(columns={0: "Legitimate", 1: "Suspicious"})
)

campaign_counts.plot(kind="bar", figsize=(12, 6))
plt.title("Figure 3: Message Counts by Campaign")
plt.xlabel("Campaign group")
plt.ylabel("Number of messages")
plt.xticks(rotation=60, ha="right")
plt.legend(title="Label")
plt.tight_layout()
plt.savefig(RESULTS_PATH / "eda_campaign_counts.png", dpi=150)
plt.close()

print("\nEDA completed!")
print("Saved visualizations:")
print(RESULTS_PATH / "eda_label_distribution.png")
print(RESULTS_PATH / "eda_channel_by_label.png")
print(RESULTS_PATH / "eda_campaign_counts.png")
