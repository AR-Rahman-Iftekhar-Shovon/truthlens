from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def summarize(y_true, y_pred, labels: list[str]) -> dict:
    return {
        "labels": labels,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro")), 4),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
