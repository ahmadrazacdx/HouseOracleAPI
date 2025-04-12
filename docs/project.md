# Pakistan Real Estate Price Prediction System

## Overview
Machine learning-powered system for property price prediction and recommendations in Pakistan's real estate market.

## Features
- Price prediction for rent/sale properties
- Similar property recommendations
- Location-based pricing tiers
- Automated feature engineering

## Tech Stack
**Backend**:
- Flask REST API
- Scikit-Learn pipelines
- Joblib model serialization

**ML**:
- XGBoost Prediction
- KNN recommendations with KMeans Clustering


## Repository Structure
```
project-root/
├── api/               # Core application logic
├── artifacts/         # Serialized models
└── docs/              # Documentation
```

## Development Setup
### Clone Repository
```python
git clone https://github.com/yourusername/property-api.git
```
### Install Dependencies
```python
pip install -r requirements.txt
```