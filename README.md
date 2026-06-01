# dust_prediction_project

Dust concentration prediction and parameter optimization using dual BP neural networks + PSO.

---

## Directory Structure

```text
dust_prediction_project/
├─ main.py                  # Entry point (train / predict modes)
├─ train.py                 # Model definitions + training pipeline
├─ predict.py               # Prediction + PSO optimization
├─ models/                  # Training artifacts
│  ├─ model_dust_fold_*.pth # Dust concentration models (5 folds)
│  ├─ model_b_fold_*.pth    # B-value models (5 folds)
│  ├─ scaler_x.pkl          # Input feature scaler
│  ├─ scaler_y.pkl          # Dust output scaler
│  └─ scaler_x_b.pkl        # B-value input scaler
└─ README.md
```

---

## Overview

| Mode | Command | Description |
|------|---------|-------------|
| Train | `python main.py --mode train` | Read CSV → data augmentation → 5-fold CV → save artifacts |
| Predict | `python main.py --mode predict` | Load models → user input → PSO optimization → output optimal parameters |

---

## Model Architecture

### Dust Concentration Prediction (BPNeuralNetwork)

```
Input(14) → fc1(64)+ReLU → Dropout(0.3) → fc2(32)+ReLU → fc3(16)+ReLU → fc4(6)
```

- Input: 7 raw features + 7 squared terms = 14 dimensions
- Output: dust concentration at 0 m / 5 m / 10 m / 15 m / 20 m / 25 m (6 positions)

### B-value Prediction (EnhancedBPNeuralNetwork)

```
Input(14) → fc1(128)+ReLU → Dropout(0.3) → fc2(64)+ReLU → fc3(32)+ReLU → fc4(1)
```

- Input: same 14 dimensions
- Output: a single B-value (absolute value, guaranteed positive)

---

## Input / Output Parameters

### Input Features (7)

| Variable | Description | Unit |
|----------|-------------|------|
| dmmj | Tunnel cross-sectional area | m² |
| yf | Press-in air volume | m³/min |
| cf | Extraction air volume | m³/min |
| xfk | Distance from exhaust inlet to tunnel face | m |
| fbft | Dust control device length | m |
| kcjl | Distance from dust control device to tunnel face | m |
| jzb | Radial-axial air volume ratio | — |

### PSO Optimization Targets (3)

| Parameter | Search Range | Order |
|-----------|-------------|-------|
| jzb (radial-axial air ratio) | 0.5 ~ 1.5 | Stage 1 |
| cf (extraction air volume) | 350 ~ 600 | Stage 2 |
| kcjl (dust control device distance) | 10 ~ 20 | Stage 3 |

### Evaluation Metrics

- **A-value**: dust concentration uniformity index (lower is better), `A = Σ(di²) / Σ(di)`
- **B-value**: single scalar predicted by the B-value network (absolute value)
- **C-value**: composite score, `C = 100 × B × 0.4 + A × 0.6`

---

## Data Pipeline (Training)

1. **Outlier removal**: rows with Z-score > 3 are discarded
2. **Feature expansion**: 7 raw features → concatenate squared terms → 14 dimensions
3. **Data augmentation**: Gaussian noise (σ=0.01) added to each sample, 5× expansion
4. **Normalization**: MinMaxScaler to [0, 1]
5. **5-fold cross-validation**: independent training per fold with early stopping (patience=200, max epochs=5000)

---

## Dependencies

Requires **Python 3.9+**.

```bash
pip install torch numpy pandas scikit-learn pyswarm
```

---

## Usage

Run all commands from the project root directory.

### 1. Training

```bash
python main.py --mode train --dust_file "粉尘数据all.csv" --b_file "粉尘数据-b.csv"
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dust_file` | 粉尘数据all.csv | Path to dust concentration CSV |
| `--b_file` | 粉尘数据-b.csv | Path to B-value CSV |
| `--save_dir` | ./models | Output directory for models and scalers |
| `--max_epochs` | 5000 | Maximum training epochs |
| `--patience` | 200 | Early stopping patience |

Generated artifacts (`models/`):
- `model_dust_fold_1~5.pth` — dust concentration model weights (5 folds)
- `model_b_fold_1~5.pth` — B-value model weights (5 folds)
- `scaler_x.pkl` / `scaler_y.pkl` / `scaler_x_b.pkl` — MinMaxScaler objects

### 2. Prediction & Optimization

```bash
python main.py --mode predict
```

Interactive workflow:

1. Enter 7 working-condition parameters
2. Initial dust concentration distribution and A/C values are computed
3. Sequential PSO optimization: radial-axial air ratio → extraction air volume → dust control device distance
4. Per-stage results and final optimal parameters are displayed

---

## FAQ

**Missing CSV files**: Training requires two data files with column names matching those defined in `train.py` (`dust_input_columns`, `dust_output_columns`, `b_input_columns`).

**Model files not found**: Run `--mode train` first to generate model artifacts, then run `--mode predict`.

**pyswarm installation fails**: `pyswarm` is a pure-Python PSO library. Use `pip install pyswarm` — no compilation required.
