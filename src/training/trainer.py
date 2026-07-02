"""Trainer: trenira model, prati validaciju, čuva najbolji po val macro F1.

Za svaku epohu: prođe kroz train batch-eve (forward -> loss -> backward -> korak
optimizatora), pa proceni na val skupu. Pamti epohu sa najboljim val macro F1 i
istoriju metrika (za figure i reproducibilnost).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()          # ponisti stare gradijente
        outputs = model(images)        # forward: predikcija
        loss = criterion(outputs, labels)  # koliko gresi
        loss.backward()                # backward: izracunaj gradijente
        optimizer.step()               # podesi tezine
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_true, all_pred = [], []
    for images, labels in tqdm(loader, desc="eval", leave=False):
        preds = model(images.to(device)).argmax(dim=1).cpu()
        all_true.extend(labels.tolist())
        all_pred.extend(preds.tolist())
    return compute_metrics(all_true, all_pred)


def fit(model, train_loader, val_loader, *, epochs, lr, device, out_dir, weight=None, patience=5):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=weight.to(device) if weight is not None else None
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best_score, best_epoch = [], -1.0, -1
    epochs_no_improve = 0
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{f"val_{k}": v for k, v in val.items()}})
        print(f"Epoha {epoch}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_macro_f1={val['macro_f1']:.4f}")

        if val["macro_f1"] > best_score:          # cuvaj NAJBOLJI po macro F1
            best_score, best_epoch = val["macro_f1"], epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:     # EARLY STOPPING
                print(f"Early stopping: nema poboljsanja {patience} epoha zaredom.")
                break

    (out_dir / "history.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_score": best_score,
                    "monitor": "macro_f1", "history": history}, indent=2),
        encoding="utf-8",
    )
    print(f"Najbolji: epoha {best_epoch}, val macro F1 = {best_score:.4f}")
    return best_score