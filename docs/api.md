# Property Price Prediction API

## Overview
REST API for Pakistan's Real Estate price prediction and recommendations. Built with Flask and machine learning pipelines.




## Endpoints

### Prediction
```http
POST /v1/predict
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

**Success Response**:
```json
{
  "predicted_price": 450000,
  "currency": "PKR",
  "timestamp": "2023-08-20T12:34:56Z"
}
```


## Error Handling
| Code | Status          | Description                     |
|------|-----------------|---------------------------------|
| 400  | Bad Request     | Invalid input parameters        |
| 401  | Unauthorized    | Missing/invalid API key         |
| 429  | Too Many Requests | Rate limit exceeded            |
| 500  | Server Error    | Internal processing error       |

Example Error Response:
```json
{
  "error": "Invalid input",
  "details": "Purpose must be rent or sale",
  "code": 400
}
```