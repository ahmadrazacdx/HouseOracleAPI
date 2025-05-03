# Pakistan Real Estate Price Prediction System

## 🚀 Overview
Machine learning-powered system for property price prediction and recommendations in Pakistan's Real Estate and Housing Data.<br>
**Dataset:** [Link](https://www.kaggle.com/datasets/jillanisofttech/pakistan-house-price-dataset)

## ✨ Features
- Price prediction for Rent/Sale Properties
- Similar Property Recommendations
- Location-Based Pricing Tiers

## 🛠️ Tech Stack
### 🎨 Backend:
- Flask REST API
- Scikit-Learn Pipelines
- Joblib Model Serialization

### ⚙️ ML:
- XGBoost Prediction 
- Scikit-Learn Nearest Neighbours (NN) Recommendations with KMeans Clustering


## 📦 Project Structure
```
HouseOrcaleAPI/
├── api/               # Core API logic        
└── artifacts/         # Serialized models
```

## ⚙️ Development Setup
### Clone Repository
```python
https://github.com/ahmadrazacdx/HouseOracleAPI.git
```
### Install Dependencies
```python
pip install -r requirements.txt
```
---
# 📖 Prediction API Docs
## Endpoints
### Home
```http
<api url>/
```
### Prediction
```http
<api url>/predict
```
**Request Body**:
```json
{
  "purpose": "rent|sale",
  "property_type":string,
  "location":string,
  "city":string,
  "province_name": string,
  "baths":[int],
  "bedrooms":[int],
  "area":[int],
  "Area Category":string
}
```
**Example Request**
```json
{
    "purpose": "rent",
    "property_type": "House",
    "location": "DHA Defence Islamabad",
    "city": "Islamabad",
    "province_name": "Islamabad Capital",
    "baths": [
        3
    ],
    "bedrooms": [
        4
    ],
    "area": [
        8
    ],
    "Area Category": "5-10 Marla"
}
```
**Success Response**:
```json
{
    "predicted_price": 47243,
    "recommendations": [
        {
            "Area Category": "1-5 Kanal",
            "area": 80.0,
            "baths": 3,
            "bedrooms": 3,
            "city": "Islamabad",
            "location": "F-7",
            "price": 200000,
            "property_type": "House",
            "province_name": "Islamabad Capital"
        },
        {
            "Area Category": "1-5 Kanal",
            "area": 72.0,
            "baths": 4,
            "bedrooms": 4,
            "city": "Islamabad",
            "location": "F-8",
            "price": 170000,
            "property_type": "House",
            "province_name": "Islamabad Capital"
        },
        {
            "Area Category": "1-5 Kanal",
            "area": 72.0,
            "baths": 4,
            "bedrooms": 4,
            "city": "Islamabad",
            "location": "E-11",
            "price": 50000,
            "property_type": "Upper Portion",
            "province_name": "Islamabad Capital"
        }
    ]
}
```


### Error Handling
| Code | Status            | Description                     |
|------|-------------------|---------------------------------|
| 400  | Bad Request       | Missing required fields        |
| 404  | Invalid Request   | Invalid purpose. Use "rent" or "sale"
| 500  | Server Error      | Prediction failed       |

**Example Error Response:**
```json
{
  "error": "Invalid input",
  "details": "Purpose must be rent or sale",
  "code": 400
}
```
## 📄 License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

## 📬 Contact
### Ahmad Raza

[![GitHub](https://img.shields.io/badge/GitHub-Profile-181717?logo=github&logoColor=fff)](https://github.com/ahmadrazacdx)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?logo=linkedin&logoColor=fff)](https://linkedin.com/in/ahmadrazacdx)  

---

⭐ If you find this project useful, please give it a star on [GitHub](https://github.com/ahmadrazacdx/HousOracleAPI)!  
