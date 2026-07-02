"""Metrike za neuravnotežene skupove.

macro F1 (PRIMARNA), uz macro Precision i Recall.
weighted F1 pondere po broju primera; balanced accuracy je accuracy prilagođen
disbalansu. Obična accuracy je sporedna (varljiva kad su klase neuravnotežene).
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),  # sporedna, zbog profesorove primedbe
    }