
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --------------------------------------------------
# 1. PATHS AND PAGE SETTINGS
# --------------------------------------------------

BASE = Path(__file__).parent
MODEL_PATH = BASE / "models" / "best_model.joblib"
DATA_PATH = BASE / "data" / "messages.csv"
RESULTS_PATH = BASE / "results"
COMPARISON_PATH = RESULTS_PATH / "model_comparison.csv"
PREDICTIONS_PATH = RESULTS_PATH / "test_predictions.csv"

st.set_page_config(
    page_title="PhishShift Message Risk Scanner",
    page_icon="🛡️",
    layout="wide",
)

# --------------------------------------------------
# 2. PROFESSIONAL WHITE THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFFFFF !important;
        color: #202124 !important;
    }

    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] {
        background-color: #F7F9FC !important;
    }

    h1, h2, h3, h4, p, label,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #202124 !important;
    }

    /* Message input: white background and readable text */
    .stTextArea textarea,
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #202124 !important;
        -webkit-text-fill-color: #202124 !important;
        caret-color: #155EEF !important;
        border: 1px solid #D0D5DD !important;
        border-radius: 8px !important;
        opacity: 1 !important;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder {
        color: #667085 !important;
        -webkit-text-fill-color: #667085 !important;
        opacity: 1 !important;
    }

    [data-baseweb="textarea"] > div,
    [data-baseweb="input"] > div {
        background-color: #FFFFFF !important;
    }

    /* Primary buttons */
    div.stButton > button {
        background-color: #155EEF !important;
        color: #FFFFFF !important;
        border: 1px solid #155EEF !important;
        border-radius: 8px !important;
        min-height: 42px;
    }

    div.stButton > button p {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }

    div.stButton > button:hover {
        background-color: #1047B8 !important;
        border-color: #1047B8 !important;
    }

    /* White chart and data containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #EAECF0 !important;
    }

    [data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. LOAD TRAINED MODEL
# --------------------------------------------------

st.title("🛡️ PhishShift")
st.subheader("Message Risk Scanner")
st.caption("Cybersecurity NLP • Message risk analysis")

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run: python train_model.py")
    st.stop()


@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


model = load_model(str(MODEL_PATH))

# --------------------------------------------------
# 4. IDENTIFY POTENTIAL RISK CUES
# --------------------------------------------------

def find_risk_cues(text):
    patterns = {
        "Urgency or pressure": (
            r"\b(urgent|immediately|act now|limited time|"
            r"as soon as possible|before access is affected)\b"
        ),
        "Account verification": (
            r"\b(verify|verification|confirm your account|"
            r"account access|access token)\b"
        ),
        "Payment or financial request": (
            r"\b(payment|invoice|bank|transaction|refund)\b"
        ),
        "Reward or prize": (
            r"\b(prize|reward|winner|claim your)\b"
        ),
        "Sensitive information": (
            r"\b(password|passcode|verification code|pin)\b"
        ),
        "Redacted or defanged URL marker": (
            r"\[url_redacted\]|hxxps?|\[\.\]|\[:\]"
        ),
    }

    return [
        cue
        for cue, pattern in patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE)
    ]


# --------------------------------------------------
# 5. SCAN HISTORY
# --------------------------------------------------

if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

st.markdown("### Scan a Message")

message = st.text_area(
    "Paste message text",
    height=150,
    placeholder="Paste the message you want to inspect here...",
    key="message_input",
)

scan_clicked = st.button(
    "Scan Message",
    type="primary",
    use_container_width=True,
)

if scan_clicked:
    if not message.strip():
        st.warning("Please enter a message before scanning.")
    else:
        try:
            suspicious_score = float(
                model.predict_proba([message.strip()])[0][1]
            )

            cues = find_risk_cues(message)

            if suspicious_score >= 0.65:
                classification = "POTENTIALLY SUSPICIOUS"
            elif suspicious_score <= 0.35:
                classification = "NO STRONG SUSPICIOUS PATTERN"
            else:
                classification = "UNCERTAIN"

            st.session_state.scan_history.append({
                "message": message.strip(),
                "score": suspicious_score,
                "classification": classification,
                "cues": cues,
            })

        except Exception as error:
            st.error(f"Unable to scan this message: {error}")


# --------------------------------------------------
# 6. DISPLAY LATEST RESULT
# --------------------------------------------------

if st.session_state.scan_history:
    latest = st.session_state.scan_history[-1]

    st.divider()
    st.markdown("## Latest Scan Result")

    score_col, result_col = st.columns(2)

    with score_col:
        st.metric(
            "Suspicious Score",
            f"{latest['score'] * 100:.1f}%",
        )
        st.progress(
            max(0.0, min(1.0, latest["score"]))
        )

    with result_col:
        st.markdown("**Classification**")

        if latest["classification"] == "POTENTIALLY SUSPICIOUS":
            st.error(latest["classification"])
        elif latest["classification"] == "UNCERTAIN":
            st.warning(latest["classification"])
        else:
            st.success(latest["classification"])

    st.markdown("### Risk Cues")

    if latest["cues"]:
        for cue in latest["cues"]:
            st.write(f"- {cue}")
    else:
        st.write("No listed risk cues were detected.")

    st.markdown("### Safe Next Steps")
    st.write("- Do not click suspicious links or open unexpected attachments.")
    st.write("- Never share passwords, PINs, or verification codes.")
    st.write("- Verify requests through a known official website or phone number.")

    st.caption(
        "Educational decision-support prototype trained on synthetic examples. "
        "The score is not a calibrated real-world probability."
    )

# --------------------------------------------------
# 7. DYNAMIC SCAN-HISTORY CHART
# --------------------------------------------------

if st.session_state.scan_history:
    st.divider()
    st.markdown("## Scan History")
    st.write("Compare the model scores for messages scanned in this session.")

    history_df = pd.DataFrame([
        {
            "Scan": f"Scan {index + 1}",
            "Suspicious Score (%)": round(item["score"] * 100, 2),
            "Classification": item["classification"],
        }
        for index, item in enumerate(st.session_state.scan_history)
    ])

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        history_df["Scan"],
        history_df["Suspicious Score (%)"],
        marker="o",
    )

    ax.set_title("Suspicious Score by Message Scan")
    ax.set_xlabel("Message Scan")
    ax.set_ylabel("Suspicious Score (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.25)

    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(history_df, use_container_width=True)

# --------------------------------------------------
# 8. FIXED MODEL EVALUATION CHARTS
# --------------------------------------------------

st.divider()
st.markdown("## Model Evaluation")

if DATA_PATH.exists():
    data = pd.read_csv(DATA_PATH)

    dataset_col, comparison_col = st.columns(2)

    # Dataset distribution
    with dataset_col:
        st.markdown("### Dataset Label Distribution")

        label_counts = data["label"].map({
            0: "Legitimate",
            1: "Suspicious",
        }).value_counts().reindex(
            ["Legitimate", "Suspicious"],
            fill_value=0,
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        label_counts.plot(kind="bar", ax=ax)

        ax.set_title("Synthetic Dataset Distribution")
        ax.set_xlabel("Message Label")
        ax.set_ylabel("Number of Messages")
        ax.tick_params(axis="x", rotation=0)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # Model comparison
    with comparison_col:
        st.markdown("### Model Comparison")

        if COMPARISON_PATH.exists():
            comparison = pd.read_csv(COMPARISON_PATH)

            metrics = [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ]

            fig, ax = plt.subplots(figsize=(8, 4))

            comparison.set_index("model")[metrics].plot(
                kind="bar",
                ax=ax,
                ylim=(0, 1.05),
            )

            ax.set_title("Held-Out Synthetic Test Results")
            ax.set_xlabel("Model")
            ax.set_ylabel("Metric Score")
            ax.tick_params(axis="x", rotation=15)
            ax.legend(loc="lower right")

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        else:
            st.info(
                "Model comparison not found. "
                "Run python train_model.py first."
            )

    # Confusion matrix
    st.markdown("### Confusion Matrix — Logistic Regression")

    if PREDICTIONS_PATH.exists():
        predictions_df = pd.read_csv(PREDICTIONS_PATH)

        actual = predictions_df["label"]
        predicted = predictions_df["logistic_regression_prediction"]

        cm = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        )

        fig, ax = plt.subplots(figsize=(6, 4))

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Legitimate", "Suspicious"],
        )

        display.plot(
            ax=ax,
            colorbar=False,
            values_format="d",
        )

        ax.set_title("Confusion Matrix — Held-Out Test Set")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.info(
            "Test predictions not found. "
            "Run python train_model.py first."
        )

    st.caption(
        "The evaluation charts show saved test-set results and do not change "
        "when individual messages are scanned. The dataset is synthetic; "
        "these results do not prove real-world detection accuracy."
    )

else:
    st.warning("Dataset not found. Check data/messages.csv.")
#streamlit run app.py