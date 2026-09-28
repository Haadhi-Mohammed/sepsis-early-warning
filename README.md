# Sepsis Early Warning System

A production-grade machine learning system that predicts sepsis onset 
6 hours before clinical diagnosis using ICU patient vitals data.

## Project Overview

Sepsis is the leading cause of hospital death globally. Early detection 
is critical — every hour of delayed treatment increases mortality by 7%. 
This system provides clinicians with a 6-hour early warning, giving 
time to intervene.

## Dataset

PhysioNet Sepsis Challenge 2019  
- 40,336 ICU patients  
- Hourly vital signs and lab values  
- Binary sepsis onset label  

## Tech Stack

| Layer | Tools |
|---|---|
| Modelling | PyTorch (LSTM) |
| Experiment tracking | MLflow |
| Explainability | SHAP |
| API | FastAPI + Docker |
| Dashboard | Streamlit |
| Drift monitoring | Evidently AI |
| CI/CD | GitHub Actions |

## Project Structure

sepsis-warning/
├── data/
│   ├── raw/          # Original dataset (not tracked by Git)
│   └── processed/    # Cleaned and engineered features
├── notebooks/        # Exploratory analysis
├── src/              # Training and preprocessing scripts
├── models/           # Saved model files
├── api/              # FastAPI application
├── dashboard/        # Streamlit dashboard
├── tests/            # Automated tests
└── reports/          # Plots and metrics

## Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[api,test]"
```

## Task and metric

This follows the PhysioNet/CinC 2019 challenge: at every hour of an ICU
stay, predict `SepsisLabel` from current and past data only. The dataset
sets that label from 6 hours before clinical (Sepsis-3) onset, so a correct
positive is a 6-hour early warning. Models are scored with the challenge's
normalised utility (1 = perfect, 0 = never alerting), reimplemented in
`src/sepsis/utility.py` and tested against the official scoring code.
PhysioNet Set A (one hospital) is used for training/validation, and Set B
(a different hospital) is the held-out test set.

## Pipeline

The training pipeline lives in the `sepsis` package (`src/sepsis/`). The
same scripts run locally and, later, as SageMaker Processing/Training jobs.

```bash
# 1. Features + labels. Set A -> patient-level train/val split, Set B -> test
python -m sepsis.prepare --raw-dir data/raw --out-dir data/processed

# 2. Train. Epoch and alert thresholds are chosen on validation only
python -m sepsis.train --data-dir data/processed --model-dir models/production

# 3. Score the held-out test set once
python -m sepsis.evaluate --model-dir models/production --data-dir data/processed --out-dir reports

# Tests and API
pytest
uvicorn api.main:app --reload
```

`notebooks/01_exploration.ipynb` holds the exploratory data analysis behind
the feature choices; its charts are in `reports/`.

## Author

Haadhi Mohammed  
MSc Data Science — Coventry University (Distinction)  
[LinkedIn](https://www.linkedin.com/in/haadhi-mohammed/)