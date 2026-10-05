# Industrial Failure ML — Part 1: Data Preparation

**EAM04DS Machine Learning | MSc Data Science & Artificial Intelligence**  
**Student:** Mohammed Ali Alshamsi

Part 1 of a three-repository postgraduate project on industrial equipment failure classification using the AI4I 2020 dataset.

## Purpose
This repository addresses the dataset challenges *before modelling*: class imbalance, target leakage, identifiers, data quality and feature construction.

## Dataset
AI4I 2020 Predictive Maintenance Dataset, UCI Machine Learning Repository. It contains 10,000 synthetic observations and 339 `Machine failure` positives (3.39%).

Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

## Leakage prevention
The target is `Machine failure`. `TWF`, `HDF`, `PWF`, `OSF` and `RNF` are excluded because they encode failure modes and would leak target information. `UDI` and `Product ID` are identifiers and are also excluded.

## Model inputs
Original operational variables: product `Type`, air/process temperature, rotational speed, torque and tool wear.

Engineered variables:
- temperature difference = process temperature − air temperature;
- power proxy = torque × rotational speed;
- torque × wear interaction.

These are engineering-inspired proxies, not direct physical power or causal claims.

## Reproduce
```bash
pip install -r requirements.txt
python src/prepare_data.py
pytest -q
```

Outputs are written to `outputs/tables/`.

## Project sequence
1. **Part 1 — Data Preparation** (this repository)
2. **Part 2 — ML Models**: baseline, Logistic Regression, Decision Tree and Random Forest
3. **Part 3 — Evaluation**: validation-selected threshold, untouched test set and interpretation

## Limitation
AI4I 2020 is synthetic and widely studied. The project therefore focuses on leakage-aware, imbalance-aware experimental design rather than claiming deployment readiness on real equipment.
