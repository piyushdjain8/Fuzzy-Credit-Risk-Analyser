import os
import pandas as pd
import streamlit as st
from fuzzy import Fuzzy4
from ml_baseline import train_ml

Fuzzy = Fuzzy4()

st.set_page_config(page_title="Fuzzy Credit Risk Analyzer", layout="wide")

DATA_PATH = "data/uci_credit_card.csv"


# ---------- UI STYLE ----------
st.markdown("""
<style>
.card {padding:16px;border-radius:14px;background:#111827;border:1px solid #2a2a40;}
.title {font-weight:600;font-size:16px;}
.value {font-size:32px;font-weight:700;}
.badge {padding:6px 10px;border-radius:10px;font-size:12px;font-weight:600;}
.badge-low {background:#0d4025;color:#d9ffe5;}
.badge-med {background:#463a09;color:#fff3c4;}
.badge-high {background:#50121c;color:#ffd9de;}
</style>
""", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown("""
<h2 style='text-align:center;'>Fuzzy Credit Risk Analyzer and Machine Learning Benchmark</h2>
<p style='text-align:center; color:#aaaaaa; margin-top:4px;'>
This dashboard estimates borrower risk using a fuzzy rule-based scoring model
and compares it with a machine-learning default probability model.
</p>
""", unsafe_allow_html=True)


# ---------- LOAD DATA ----------
@st.cache_data
def load_df():
    if not os.path.exists(DATA_PATH):
        st.warning(
            "Dataset not found.\n\n"
            "Download the dataset from Kaggle and place it at:\n"
            "`data/uci_credit_card.csv`\n\n"
            "Dataset link:\n"
            "https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset"
        )
        st.stop()

    df = pd.read_csv(DATA_PATH)
    df = df.loc[:, ~df.columns.duplicated()]

    df = df.rename(columns={
        "LIMIT_BAL": "Credit Limit",
        "PAY_0": "Recent Repayment Status",
        "BILL_AMT1": "Last Billed Amount",
        "PAY_AMT1": "Last Payment Amount",
        "default.payment.next.month": "Did Customer Default Next Month"
    })

    df["Recent Repayment Status"] = df["Recent Repayment Status"].fillna(0)
    df["Last Payment Amount"] = df["Last Payment Amount"].fillna(0)

    return df


df = load_df()


# ---------- TRAIN MODEL ----------
@st.cache_resource
def get_model():
    mdf = df.rename(columns={
        "Credit Limit": "LIMIT_BAL",
        "Recent Repayment Status": "PAY_0",
        "Last Billed Amount": "BILL_AMT1",
        "Last Payment Amount": "PAY_AMT1",
        "Did Customer Default Next Month": "default.payment.next.month"
    })
    return train_ml(mdf)


model, scaler, metrics, auc = get_model()


# ---------- KPI + METRIC EXPLANATION ----------
c1, c2 = st.columns([1, 2])

with c1:
    st.markdown(
        "<div class='card'><div class='title'>ROC-AUC (Model Performance)</div>"
        f"<div class='value'>{auc:.3f}</div></div>",
        unsafe_allow_html=True
    )

with c2:
    st.markdown("<div class='title'>Classification Metrics</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:13px; color:#bbbbbb; margin-bottom:6px;'>
    These metrics describe how well the machine-learning model identifies customers who default on payments.
    </div>
    <ul style='font-size:13px; color:#cccccc;'>
        <li><b>Precision</b> – Of the customers predicted as defaulters, how many actually defaulted.</li>
        <li><b>Recall</b> – Of the customers who truly defaulted, how many the model successfully identified.</li>
        <li><b>F1 Score</b> – A balanced measure combining Precision and Recall.</li>
        <li><b>Support</b> – The number of samples in each class.</li>
        <li><b>Class 0</b> – Customers who did not default.</li>
        <li><b>Class 1</b> – Customers who defaulted.</li>
    </ul>
    """, unsafe_allow_html=True)

    mdf = pd.DataFrame(metrics).T.reset_index()
    mdf.columns = ["Class", "Precision", "Recall", "F1 Score", "Samples"]
    st.dataframe(mdf, use_container_width=True)

st.markdown("---")


# ---------- TWO-PANEL LAYOUT ----------
left, right = st.columns(2)


# ===================== EXISTING RECORD MODE =====================
with left:
    st.markdown("### Evaluate Existing Customer")

    idx = st.number_input("Select Customer Record", 0, len(df) - 1, 0)
    rec = df.iloc[int(idx)]

    st.markdown("#### Customer Details")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.table(pd.DataFrame({
        "Field": ["Credit Limit", "Recent Repayment Status", "Last Billed Amount", "Last Payment Amount"],
        "Value": [
            rec["Credit Limit"],
            rec["Recent Repayment Status"],
            rec["Last Billed Amount"],
            rec["Last Payment Amount"]
        ]
    }))

    st.markdown("</div>", unsafe_allow_html=True)

    vals = [
        float(rec["Credit Limit"]),
        float(rec["Recent Repayment Status"]),
        float(rec["Last Billed Amount"]),
        float(rec["Last Payment Amount"])
    ]

    fuzzy = Fuzzy.score(*vals)
    ml = model.predict_proba(scaler.transform([vals]))[0][1]

    if fuzzy < 35:
        level = "Low Risk"; css = "low"
    elif fuzzy < 65:
        level = "Medium Risk"; css = "med"
    else:
        level = "High Risk"; css = "high"

    st.markdown("#### Risk Assessment")

    r1, r2 = st.columns(2)

    r1.markdown(
        f"<div class='card'><div class='title'>Fuzzy Risk Score</div>"
        f"<div class='value'>{fuzzy:.2f}</div>"
        f"<span class='badge badge-{css}'>{level}</span></div>",
        unsafe_allow_html=True
    )

    r2.markdown(
        f"<div class='card'><div class='title'>ML Default Probability</div>"
        f"<div class='value'>{ml:.3f}</div></div>",
        unsafe_allow_html=True
    )


# ===================== MANUAL INPUT MODE =====================
with right:
    st.markdown("### Manually Evaluate a Customer")

    c1, c2 = st.columns(2)

    credit = c1.number_input("Credit Limit", 0, 2_000_000, 200000)
    repay  = c1.number_input("Recent Repayment Status (-2 to 8)", -2, 8, 0)
    bill   = c2.number_input("Last Billed Amount", 0, 1_000_000, 50000)
    pay    = c2.number_input("Last Payment Amount", 0, 500000, 10000)

    if st.button("Compute Risk", use_container_width=True):
        fuzzy = Fuzzy.score(credit, repay, bill, pay)
        ml = model.predict_proba(scaler.transform([[credit, repay, bill, pay]]))[0][1]

        if fuzzy < 35:
            level = "Low Risk"; css = "low"
        elif fuzzy < 65:
            level = "Medium Risk"; css = "med"
        else:
            level = "High Risk"; css = "high"

        st.markdown("#### Risk Assessment")

        r1, r2 = st.columns(2)

        r1.markdown(
            f"<div class='card'><div class='title'>Fuzzy Risk Score</div>"
            f"<div class='value'>{fuzzy:.2f}</div>"
            f"<span class='badge badge-{css}'>{level}</span></div>",
            unsafe_allow_html=True
        )

        r2.markdown(
            f"<div class='card'><div class='title'>ML Default Probability</div>"
            f"<div class='value'>{ml:.3f}</div></div>",
            unsafe_allow_html=True
        )
