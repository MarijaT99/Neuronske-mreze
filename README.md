# Hijerarhijska klasifikacija biljnih bolesti korišćenjem dubokih neuronskih mreža

Master projekat iz predmeta *Neuronske mreže*, FTN, Univerzitet u Novom Sadu.
Autor: **Marija Tadić** (E9 12/2024).

Klasifikacija biljnih bolesti sa fotografija listova (dataset **PlantVillage**),
poređenje *flat* klasifikatora (svih 38 klasa odjednom) i **hijerarhijskog**
pristupa (prvo se prepozna **vrsta** biljke, zatim **bolest** unutar te vrste).

---

## 1. Ideja i pristup

Kompleksan zadatak (38 klasa = vrsta + bolest) deli se na dva jednostavnija koraka:

1. **Nivo 1 - vrsta biljke** (14 klasa): jedan model prepoznaje o kojoj je biljci reč.
2. **Nivo 2 - bolest** (unutar vrste): za svaku vrstu sa više od jedne bolesti postoji
   poseban model koji bira bolest. Vrste sa samo jednom klasom
   (*Orange, Soybean, Raspberry, Squash, Blueberry* - sve „healthy") nemaju model
   na drugom nivou. Ukupno **9 disease modela**.

Kao referenca (baseline) trenira se prost CNN „od nule", a kao napredni pristup
koristi se **transfer learning** (ResNet-50 i EfficientNet-B0). Meri se i
**propagacija greške** kroz hijerarhiju (koliko grešaka potiče od pogrešne vrste).

---

## 2. Struktura projekta

```
Neuronske-mreze/
├── data/
│   └── splits/                 # stratifikovana podela (već u repo-u)
│       ├── all.csv, train.csv, val.csv, test.csv
│       └── label_maps.json     # mapiranja klasa/vrsta/bolesti
├── src/
│   ├── data/                   # učitavanje podataka, podele, transformacije
│   ├── models/                 # baseline CNN, transfer (ResNet-50/EfficientNet), hijerarhija
│   ├── training/               # trening petlja (Adam, early stopping, čuva najbolji)
│   └── evaluation/             # metrike (macro F1 primarna)
├── scripts/                    # skripte za pokretanje (vidi ispod)
├── experiments/                # rezultati treninga (history.json; .pt težine se NE čuvaju u git-u)
├── results/                    # figure i tabele (konfuziona matrica, per-class F1)
├── requirements.txt
└── README.md
```

---

## 3. Skup podataka (PlantVillage)

- **54.305** RGB slika listova, **38 klasa** (vrsta + bolest), **14 vrsta**.
- Podela: **70% / 15% / 15%** (train / val / test), stratifikovano, `seed=42`.
- Podele su **već u repo-u** (`data/splits/`), pa nije obavezno ponovo ih praviti.

Skup je javno dostupan na Kaggle-u:
[PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
(koristi se `color` verzija).

U komandama ispod `DATA_ROOT` je putanja do foldera `color`:
- Na Kaggle-u: `/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color`
- Lokalno: putanja do raspakovanog `color` foldera na tvom računaru.

---

## 4. Zahtevi i instalacija

Potreban je **Python 3.10+**. Preporučuje se virtuelno okruženje.

### Windows (PowerShell)
```powershell
git clone https://github.com/MarijaT99/Neuronske-mreze.git
cd Neuronske-mreze
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
> Ako aktivacija javi grešku o „execution policy", pokreni jednom:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

### Linux / macOS
```bash
git clone https://github.com/MarijaT99/Neuronske-mreze.git
cd Neuronske-mreze
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Glavne zavisnosti (pun spisak u `requirements.txt`): PyTorch, torchvision, timm,
albumentations, scikit-learn, pandas, numpy, matplotlib, seaborn.

---

## 5. Pokretanje - korak po korak

> Napomena o uređaju: dodaj `--device cuda` ako imaš GPU, u suprotnom `--device cpu`
> (podrazumevano je `cpu`). Za brzu probu na CPU dodaj `--limit 200` (uzme mali uzorak).

### (Opciono) 5.0 Ponovno pravljenje podela
Podele su već u repo-u; ovaj korak je potreban samo ako ih praviš iznova:
```bash
python scripts/prepare_splits.py --data-root "DATA_ROOT" --out-dir data/splits --seed 42
```

### 5.1 Treniranje FLAT modela (svih 38 klasa odjednom)
```bash
# Baseline CNN (od nule)
python scripts/train_flat.py --data-root "DATA_ROOT" --model baseline_cnn    --epochs 25 --device cuda

# ResNet-50 (transfer learning)
python scripts/train_flat.py --data-root "DATA_ROOT" --model resnet50        --epochs 8  --device cuda

# EfficientNet-B0 (transfer learning)
python scripts/train_flat.py --data-root "DATA_ROOT" --model efficientnet_b0 --epochs 8  --device cuda
```

### 5.2 Treniranje HIJERARHIJE (ResNet-50)
```bash
# Nivo 1 - vrsta (14 klasa)
python scripts/train_species.py --data-root "DATA_ROOT" --model resnet50 --epochs 10 --device cuda

# Nivo 2 - bolest po vrsti (trenira svih 9 disease modela)
python scripts/train_disease.py --data-root "DATA_ROOT" --model resnet50 --epochs 12 --device cuda
```

### 5.3 Evaluacija na TEST skupu
Računa test metrike za **sve flat modele** i **hijerarhiju**, pravi konfuzionu
matricu i per-class F1:
```bash
python scripts/evaluate.py --data-root "DATA_ROOT" --device cuda
```
Rezultati:
- `experiments/resnet50_hierarchical/test_metrics.json` (sve test metrike)
- `results/confusion_flat.png`, `results/per_class_f1.csv`

### 5.4 Figure za rad
```bash
python scripts/make_figures.py
```
Pravi PNG figure u `results/` (krive treninga, poređenje modela, per-class F1,
propagacija greške). Ne zahteva GPU ni podatke - čita samo sačuvane rezultate.

### (Opciono) Sanity provere
```bash
python scripts/sanity_data.py     # provera podela (curenje podataka, brojevi)
python scripts/sanity_model.py    # provera da modeli rade (forward prolaz)
```

---

## 6. Reprodukcija na Kaggle-u (GPU)

Trening je rađen na Kaggle GPU-u. Ceo postupak (klon repo-a → trening svih modela →
evaluacija) izvodi se u notebooku sa 3 ćelije:

```python
# Ćelija 1: kod
!git clone https://github.com/MarijaT99/Neuronske-mreze.git
%cd Neuronske-mreze

# Ćelija 2: podaci
DATA_ROOT = "/kaggle/input/datasets/abdallahalidev/plantvillage-dataset/color"

# Ćelija 3: trening + evaluacija
!python scripts/train_flat.py    --data-root "{DATA_ROOT}" --model resnet50        --epochs 8  --device cuda --num-workers 4
!python scripts/train_flat.py    --data-root "{DATA_ROOT}" --model efficientnet_b0 --epochs 8  --device cuda --num-workers 4
!python scripts/train_flat.py    --data-root "{DATA_ROOT}" --model baseline_cnn    --epochs 25 --device cuda --num-workers 4
!python scripts/train_species.py --data-root "{DATA_ROOT}" --model resnet50        --epochs 10 --device cuda --num-workers 4
!python scripts/train_disease.py --data-root "{DATA_ROOT}" --model resnet50        --epochs 12 --device cuda --num-workers 4
!python scripts/evaluate.py      --data-root "{DATA_ROOT}" --device cuda
```

---

## 7. Evaluacione metrike

Skup je **nebalansiran**, pa je **Accuracy neprikladna** kao glavna metrika.
Primarna metrika je **macro F1** (jednaka težina svakoj klasi), uz **Precision**,
**Recall** i **weighted F1**. Accuracy se navodi samo informativno.
Greške se analiziraju **konfuzionom matricom** i **per-class F1**, a kod hijerarhije
i **propagacijom greške** (udeo grešaka koje potiču od pogrešno prepoznate vrste).

---

## 8. Rezultati

Kompletne test metrike su u `experiments/resnet50_hierarchical/test_metrics.json`,
a figure u `results/`. Hijerarhijski pristup nadmašuje flat baseline po macro F1,
uz gotovo savršeno prepoznavanje vrste na prvom nivou. Detaljna tabela poređenja i
diskusija nalaze se u radu (`paper/`).

---

## 9. Napomena o težinama modela

Naučene težine (`best_model.pt`, ~90–100 MB po modelu) se **ne čuvaju u git-u**
(navedene su u `.gitignore`). Da bi ih dobio, pokreni trening (poglavlje 5) ili
reprodukuj na Kaggle-u (poglavlje 6).