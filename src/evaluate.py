import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, roc_auc_score
)

os.makedirs("reports/figures", exist_ok=True)

LABELS = {
    "lr_weighted": "LR (class weights)",
    "rf_weighted": "RF (class weights)",
    "rf_smote":    "RF (SMOTE)",
    "xgb":         "XGBoost (scale_pos_weight)",
}

def best_threshold(precision, recall, thresholds):
    """F1-maximizing threshold."""
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-9)
    return thresholds[np.argmax(f1)]

def evaluate_all():
    X_test, y_test = joblib.load("models/test_data.pkl")

    plt.figure(figsize=(8, 6))
    results = []

    for name, label in LABELS.items():
        model = joblib.load(f"models/{name}.pkl")
        proba = model.predict_proba(X_test)[:, 1]

        ap = average_precision_score(y_test, proba)
        roc = roc_auc_score(y_test, proba)

        precision, recall, thresholds = precision_recall_curve(y_test, proba)
        plt.plot(recall, precision, label=f"{label} (AP={ap:.3f})")

        thresh = best_threshold(precision, recall, thresholds)
        preds = (proba >= thresh).astype(int)
        report = classification_report(y_test, preds, output_dict=True)
        fraud = report["1"]

        results.append({
            "Model": label,
            "Threshold": round(thresh, 3),
            "Precision": round(fraud["precision"], 3),
            "Recall":    round(fraud["recall"], 3),
            "F1":        round(fraud["f1-score"], 3),
            "PR-AUC":    round(ap, 3),
            "ROC-AUC":   round(roc, 3),
        })

    # PR curve
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("reports/figures/pr_curve.png", dpi=150)
    plt.close()

    # Confusion matrix for best model by PR-AUC
    best = max(results, key=lambda r: r["PR-AUC"])
    best_name = [k for k, v in LABELS.items() if v == best["Model"]][0]
    best_model = joblib.load(f"models/{best_name}.pkl")
    proba = best_model.predict_proba(X_test)[:, 1]
    preds = (proba >= best["Threshold"]).astype(int)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Genuine", "Fraud"],
                yticklabels=["Genuine", "Fraud"])
    plt.title(f"Confusion Matrix — {best['Model']}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("reports/figures/confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance (XGBoost)
    xgb = joblib.load("models/xgb.pkl")
    importances = xgb.feature_importances_
    features = X_test.columns
    top_idx = np.argsort(importances)[-15:]
    plt.figure(figsize=(7, 5))
    plt.barh(features[top_idx], importances[top_idx])
    plt.title("XGBoost — Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png", dpi=150)
    plt.close()

    print("\nResults on held-out test set:")
    print(f"{'Model':<30} {'Thresh':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'PR-AUC':>8} {'ROC-AUC':>9}")
    print("-" * 80)
    for r in results:
        print(f"{r['Model']:<30} {r['Threshold']:>7} {r['Precision']:>7} {r['Recall']:>7} {r['F1']:>7} {r['PR-AUC']:>8} {r['ROC-AUC']:>9}")

    print(f"\nBest model: {best['Model']} | Threshold: {best['Threshold']} | Recall: {best['Recall']} | Precision: {best['Precision']}")
    print("Figures saved to reports/figures/")

if __name__ == "__main__":
    evaluate_all()
