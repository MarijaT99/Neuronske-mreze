"""Pravljenje figura za rad iz sačuvanih rezultata (bez podataka i GPU-a).

Čita samo experiments/**/history.json, experiments/.../test_metrics.json i
results/per_class_f1.csv, pa upisuje PNG figure u results/.

Pokretanje:
    python scripts/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

EXP = ROOT / "experiments"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# Lepa imena flat modela (ključ = ime eksperimenta / modela)
DISPLAY = {
    "baseline_cnn": "Baseline CNN",
    "resnet50": "ResNet-50",
    "efficientnet_b0": "EfficientNet-B0",
}
# Flat modeli: lepo ime -> folder eksperimenta (za krive treninga)
FLAT_HIST = {
    "Baseline CNN": "baseline_cnn_flat",
    "ResNet-50": "resnet50_flat",
    "EfficientNet-B0": "efficientnet_b0_flat",
}


def load_history(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    return pd.DataFrame(obj["history"]), obj.get("best_epoch")


def load_test_metrics():
    return json.loads((EXP / "resnet50_hierarchical/test_metrics.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Figura 1: krive treninga flat modela (val macro F1 + train loss)
# ---------------------------------------------------------------------------
def fig_training_curves():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for naziv, folder in FLAT_HIST.items():
        path = EXP / folder / "history.json"
        if not path.exists():
            continue
        df, best = load_history(path)
        ax1.plot(df["epoch"], df["val_macro_f1"], marker="o", ms=3, label=naziv)
        ax2.plot(df["epoch"], df["train_loss"], marker="o", ms=3, label=naziv)
        if best is not None:
            row = df[df["epoch"] == best]
            if not row.empty:
                ax1.scatter([best], [row["val_macro_f1"].iloc[0]], s=80,
                            facecolors="none", edgecolors="red", zorder=5)

    ax1.set_title("Validacioni macro F1 po epohi")
    ax1.set_xlabel("epoha"); ax1.set_ylabel("macro F1")
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.set_title("Trening gubitak (loss) po epohi")
    ax2.set_xlabel("epoha"); ax2.set_ylabel("train loss")
    ax2.grid(True, alpha=0.3); ax2.legend()

    fig.suptitle("Krive treninga flat modela (crveni krug = najbolja epoha)")
    fig.tight_layout()
    out = RESULTS / "training_curves_flat.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Sacuvano:", out)


# ---------------------------------------------------------------------------
# Figura 2: poređenje modela na TEST skupu (macro F1)
# ---------------------------------------------------------------------------
def fig_model_comparison():
    tm = load_test_metrics()
    flat = tm.get("flat_models", {})
    hier = tm["hierarchical"]

    labele, vrednosti = [], []
    for key in ("baseline_cnn", "resnet50", "efficientnet_b0"):
        if key in flat:
            labele.append(DISPLAY[key])
            vrednosti.append(flat[key]["macro_f1"])
    labele.append("Hijerarhijski")
    vrednosti.append(hier["end_to_end_macro_f1"])

    boje = ["#7f8c8d"] * (len(labele) - 1) + ["#27ae60"]  # hijerarhijski istaknut
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bars = ax.bar(labele, vrednosti, color=boje, width=0.6)
    for b, v in zip(bars, vrednosti):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylim(0.90, 1.0)
    ax.set_ylabel("macro F1 (test)")
    ax.set_title("Poređenje modela na test skupu (primarna metrika: macro F1)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "model_comparison_test.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Sacuvano:", out)


# ---------------------------------------------------------------------------
# Figura 3: per-class F1 (sortirano rastuće)
# ---------------------------------------------------------------------------
def fig_per_class_f1():
    df = pd.read_csv(RESULTS / "per_class_f1.csv")
    df = df.sort_values("f1", ascending=True).reset_index(drop=True)
    labele = [c.replace("___", " – ").replace("_", " ") for c in df["class"]]
    boje = ["#c0392b" if v < 0.95 else "#2e86c1" for v in df["f1"]]

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(range(len(df)), df["f1"], color=boje)
    ax.set_yticks(range(len(df))); ax.set_yticklabels(labele, fontsize=7)
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("F1 (test)")
    ax.set_title("F1 po klasi (flat model, test)\ncrveno: F1 < 0.95")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "per_class_f1_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Sacuvano:", out)


# ---------------------------------------------------------------------------
# Figura 4: propagacija greške (vrsta vs bolest)
# ---------------------------------------------------------------------------
def fig_error_propagation():
    h = load_test_metrics()["hierarchical"]
    total = h["errors_total"]
    iz_vrste = h["errors_from_species"]
    iz_bolesti = total - iz_vrste

    kategorije = ["Greška vrste\n(nivo 1)", "Greška bolesti\n(nivo 2)"]
    vrednosti = [iz_vrste, iz_bolesti]
    boje = ["#c0392b", "#f39c12"]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(kategorije, vrednosti, color=boje, width=0.55)
    for b, v in zip(bars, vrednosti):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3,
                f"{v}\n({v / total * 100:.0f}%)", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("broj grešaka")
    ax.set_ylim(0, total)
    ax.set_title(f"Poreklo grešaka hijerarhijskog modela\n(ukupno {total} grešaka na test skupu)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = RESULTS / "error_propagation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Sacuvano:", out)


# ---------------------------------------------------------------------------
# Figura 5: najbolji val macro F1 disease modela po vrsti
# ---------------------------------------------------------------------------
def fig_disease_models():
    disease_dir = EXP / "resnet50_hierarchical/disease"
    redovi = []
    for path in sorted(disease_dir.glob("*/history.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        redovi.append((path.parent.name.replace("_", " "), obj.get("best_score", 0.0)))
    if not redovi:
        print("Nema disease modela, preskačem figuru 5.")
        return

    redovi.sort(key=lambda r: r[1])
    naziv = [r[0] for r in redovi]
    vals = [r[1] for r in redovi]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(range(len(naziv)), vals, color="#27ae60")
    ax.set_yticks(range(len(naziv))); ax.set_yticklabels(naziv, fontsize=9)
    ax.set_xlim(0.95, 1.0)
    ax.set_xlabel("najbolji validacioni macro F1")
    ax.set_title("Disease modeli po vrsti (drugi nivo hijerarhije)")
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v + 0.0005, i, f"{v:.4f}", va="center", fontsize=8)
    fig.tight_layout()
    out = RESULTS / "disease_models_f1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("Sacuvano:", out)


def main():
    fig_training_curves()
    fig_model_comparison()
    fig_per_class_f1()
    fig_error_propagation()
    fig_disease_models()
    print("\nSve figure su u:", RESULTS)


if __name__ == "__main__":
    main()