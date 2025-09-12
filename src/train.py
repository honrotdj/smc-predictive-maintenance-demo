import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
from joblib import dump

# Load our CSV into X (features) and y (labels)
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[["pressure", "temperature", "vibration", "cycle_count"]].values
    y = df["fault"].values
    return X, y

# Simple pipeline: scale features -> logistic regression
def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

# Draw and save ROC curve
def plot_roc(y_true, y_proba, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression ROC")
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    return roc_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sample.csv")
    parser.add_argument("--model_out", default="model.joblib")
    parser.add_argument("--fig_out", default="reports/figures/roc.png")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X, y = load_data(args.data)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model = build_model()
    model.fit(X_tr, y_tr)

    y_proba = model.predict_proba(X_te)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    acc = accuracy_score(y_te, y_pred)
    roc_auc = plot_roc(y_te, y_proba, args.fig_out)

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix")
    plt.savefig("reports/figures/cm.png", dpi=160, bbox_inches="tight")
    plt.close()

    

    dump(model, args.model_out)

    print("== Training Summary ==")
    print(f"Train size: {len(X_tr):,} | Test size: {len(X_te):,}")
    print(f"Accuracy: {acc:.3f}")
    print(f"ROC AUC:  {roc_auc:.3f}")
    print(f"Saved model -> {args.model_out}")
    print(f"Saved ROC   -> {args.fig_out}")

if __name__ == "__main__":
    main()

