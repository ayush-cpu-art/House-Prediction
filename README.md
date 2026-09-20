# 🏠 House Price Prediction

An end-to-end machine learning project for predicting house prices using property characteristics such as area, bedrooms, bathrooms, stories, parking, amenities, and furnishing status.

The project covers the complete ML workflow from data preprocessing and feature engineering to model comparison, hyperparameter tuning, model evaluation, and deployment.

## 🚀 Project Overview

The objective of this project is to build a regression model capable of estimating house prices from property features.

Rather than relying on a single algorithm, multiple regression models were trained and evaluated using the same dataset to compare their performance.

The project includes:

- Exploratory Data Analysis
- Data cleaning and preprocessing
- Feature engineering
- Categorical encoding
- Feature scaling
- Multiple regression models
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Feature importance analysis
- Model serialization
- Web application deployment

## 📊 Dataset

The project uses a housing dataset containing **545 observations and 13 original features**.

### Features

| Feature | Description |
|---|---|
| `area` | Area of the property |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of stories |
| `mainroad` | Whether the property is connected to the main road |
| `guestroom` | Whether a guest room is available |
| `basement` | Whether the property has a basement |
| `hotwaterheating` | Whether hot water heating is available |
| `airconditioning` | Whether air conditioning is available |
| `parking` | Number of parking spaces |
| `prefarea` | Whether the property is located in a preferred area |
| `furnishingstatus` | Furnishing status |
| `price` | Target house price |

## 🔄 Machine Learning Pipeline

```text
Raw Housing Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Categorical Encoding
        ↓
Feature Engineering
        ↓
Feature Scaling
        ↓
Train / Test Split
        ↓
Multiple Regression Models
        ↓
Model Evaluation
        ↓
Hyperparameter Tuning
        ↓
Final Model
        ↓
Model Serialization
        ↓
Web Application
