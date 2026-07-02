"""Hijerarhijski klasifikator: model za vrstu (nivo 1) + modeli za bolest po vrsti (nivo 2).

Pri predikciji: prvo se predvidi VRSTA, pa se bolest predviđa 'disease' modelom
BAŠ TE predviđene vrste. Ako vrsta bude pogrešna, i bolest je (skoro uvek) pogrešna
- to je 'propagacija greške' koju hijerarhijska evaluacija meri.

Vrste sa samo jednom klasom (npr. Orange, Soybean, Raspberry, Squash, Blueberry -
samo 'healthy') nemaju poseban disease model: njihova bolest je uvek id 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalClassifier(nn.Module):
    def __init__(self, species_model, disease_models: dict, num_species: int):
        super().__init__()
        self.species_model = species_model
        # ModuleDict trazi string kljuceve; cuvamo disease model po vrsti.
        self.disease_models = nn.ModuleDict({str(k): m for k, m in disease_models.items()})
        self.num_species = num_species

    def has_disease_head(self, species_id: int) -> bool:
        return str(species_id) in self.disease_models

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        species_logits = self.species_model(x)
        species_pred = species_logits.argmax(dim=1)          # nivo 1: koja vrsta

        disease_pred = torch.zeros_like(species_pred)
        for sid in species_pred.unique().tolist():
            if not self.has_disease_head(sid):
                continue  # trivijalna vrsta -> bolest ostaje 0
            mask = species_pred == sid                        # svi primeri te vrste
            head = self.disease_models[str(sid)]
            logits = head(x[mask])                            # nivo 2: bolest u toj vrsti
            disease_pred[mask] = logits.argmax(dim=1)

        return species_pred, disease_pred