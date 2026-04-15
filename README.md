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
pip install -r requirements.txt
```

## Author

Haadhi Mohammed  
MSc Data Science — Coventry University (Distinction)  
[LinkedIn](https://www.linkedin.com/in/haadhi-mohammed/)