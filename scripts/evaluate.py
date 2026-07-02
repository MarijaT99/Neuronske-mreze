"""Evaluacija na TEST skupu - flat model i hijerarhija (end-to-end).

Učitava najbolje sačuvane modele (best_model.pt), računa test metrike, matricu
konfuzije i per-class F1. Rezultati -> results/ i experiments/.../test_metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from src.data.loaders import build_dataset, build_loader  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.models.hierarchical import HierarchicalClassifier  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load_maps():
    m = json.loads((ROOT / "data/splits/label_maps.json").read_text(encoding="utf-8"))
    id2species = {v: k for k, v in m["species_to_id"].items()}
    id2class = {v: k for k, v in m["class_to_id"].items()}
    id2disease = {s: {v: k for k, v in dd.items()} for s, dd in m["disease_in_species_to_id"].items()}
    return m, id2species, id2class, id2disease


def load_model(name, num_classes, path, device):
    model = build_model(name, num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def evaluate_flat(data_root, device):
    ds = build_dataset(data_root, "data/splits", "test", target="flat")
    loader = build_loader(ds, batch_size=64, num_workers=2)
    model = load_model("resnet50", 38, "experiments/resnet50_flat/best_model.pt", device)

    y_true, y_pred = [], []
    for x, y in loader:
        y_pred += model(x.to(device)).argmax(1).cpu().tolist()
        y_true += y.tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    metrics = compute_metrics(y_true, y_pred)
    print("FLAT test:", {k: round(v, 4) for k, v in metrics.items()})

    _, _, id2class, _ = load_maps()
    per = f1_score(y_true, y_pred, average=None, zero_division=0)
    pd.DataFrame({"class": [id2class[i] for i in range(38)], "f1": per}).to_csv(
        RESULTS / "per_class_f1.csv", index=False, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues"); plt.colorbar()
    plt.title("Konfuziona matrica (flat, test)")
    plt.xlabel("predvidjeno"); plt.ylabel("stvarno")
    plt.savefig(RESULTS / "confusion_flat.png", dpi=120, bbox_inches="tight"); plt.close()
    return metrics


@torch.no_grad()
def evaluate_hierarchical(data_root, device):
    m, id2species, id2class, id2disease = load_maps()
    species_model = load_model("resnet50", 14,
                               "experiments/resnet50_hierarchical/species/best_model.pt", device)
    disease_models = {}
    for species, sid in m["species_to_id"].items():
        n = len(m["disease_in_species_to_id"][species])
        if n < 2:
            continue
        disease_models[sid] = load_model(
            "resnet50", n, f"experiments/resnet50_hierarchical/disease/{species}/best_model.pt", device)

    clf = HierarchicalClassifier(species_model, disease_models, num_species=14).to(device).eval()

    test_df = load_split("data/splits", "test").reset_index(drop=True)
    ds = build_dataset(data_root, "data/splits", "test", target="flat")
    loader = build_loader(ds, batch_size=64, num_workers=2)  # shuffle=False -> redosled ocuvan

    sp_all, dis_all = [], []
    for x, _ in loader:
        sp, dis = clf.predict(x.to(device))
        sp_all += sp.cpu().tolist(); dis_all += dis.cpu().tolist()
    sp_all, dis_all = np.array(sp_all), np.array(dis_all)

    true_class = test_df["class_id"].to_numpy()
    true_species = test_df["species_id"].to_numpy()
    pred_class = np.array([
        m["class_to_id"][f"{id2species[sp]}___{id2disease[id2species[sp]][dl]}"]
        for sp, dl in zip(sp_all, dis_all)
    ])

    e2e = compute_metrics(true_class, pred_class)
    sp_metrics = compute_metrics(true_species, sp_all)
    wrong = pred_class != true_class
    errors = int(wrong.sum())
    errors_from_species = int((wrong & (sp_all != true_species)).sum())

    result = {
        "end_to_end_macro_f1": e2e["macro_f1"],
        "end_to_end_weighted_f1": e2e["weighted_f1"],
        "end_to_end_accuracy": e2e["accuracy"],
        "species_macro_f1": sp_metrics["macro_f1"],
        "species_correct_frac": float((sp_all == true_species).mean()),
        "errors_total": errors,
        "errors_from_species": errors_from_species,
        "share_of_errors_due_to_species": (errors_from_species / errors if errors else 0.0),
    }
    print("HIJERARHIJA test:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.items()})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    flat = evaluate_flat(args.data_root, args.device)
    hier = evaluate_hierarchical(args.data_root, args.device)

    out = ROOT / "experiments/resnet50_hierarchical/test_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"flat": flat, "hierarchical": hier}, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print("\nSacuvano:", out)


if __name__ == "__main__":
    main()