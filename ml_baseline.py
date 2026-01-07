# ml_baseline.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = ["LIMIT_BAL", "PAY_0", "BILL_AMT1", "PAY_AMT1"]
TARGET = "default.payment.next.month"


def train_ml(df):
    """
    Stable ML benchmark pipeline.
    Ensures y is strictly 1-D even if CSV contains duplicate target columns.
    """
    df = df.copy()

    # remove duplicate column names if present
    df = df.loc[:, ~df.columns.duplicated()]

    X = df[FEATURES]

    # y may accidentally load as 2-column dataframe — fix to 1-D
    y = df[[TARGET]]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    y = y.astype(int).values.ravel()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(Xtr, ytr)

    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    metrics = classification_report(
        yte, y_pred, digits=3, output_dict=True
    )
    auc = roc_auc_score(yte, y_prob)

    return model, scaler, metrics, auc
