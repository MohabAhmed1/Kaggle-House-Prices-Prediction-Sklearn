# House Prices Prediction – Advanced Regression Techniques

Machine learning pipeline for predicting house prices using **Random Forest, XGBoost, and Ensemble Learning** on the Kaggle [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset.

**Kaggle Public Score:** `0.13212`  

---

## 📁 Folder Structure


Kaggle-House-Prices-Prediction-Using-Sklearn
│
├── data/ # Kaggle dataset
│ ├── train.csv
│ └── test.csv
│
├── notebooks/ # Training and analysis notebooks
│ └── house_price_model.ipynb
│
├── model/ # Saved trained model
│ └── house_price_model.h5
│
├── submission/ # Kaggle submission files
│ └── submission.csv
│
├── requirements.txt # Project dependencies
└── README.md # Project documentation


---

## 📌 Project Overview

This repository contains a **full machine learning pipeline** to predict residential house prices.  
It demonstrates:

- Advanced **data preprocessing** for tabular datasets
- **Model benchmarking** with multiple regression algorithms
- **Custom weighted ensemble** to achieve high precision
- Automated **Kaggle submission generation**

---

## 📊 Dataset

The dataset comes from the Kaggle competition **House Prices: Advanced Regression Techniques** and contains **79 explanatory variables** describing residential homes in Ames, Iowa.

- **train.csv** – used for model training and validation  
- **test.csv** – used to generate final Kaggle predictions

---

## ⚙️ Machine Learning Pipeline

### 🗂 1. Data Preprocessing

To ensure the models receive clean, standardized data, the following steps were implemented:

- **Target Transformation:** Applied `np.log1p` to the `SalePrice` to handle skewness.  
- **Categorical Encoding:** Used One-Hot Encoding (`pd.get_dummies`) for categorical variables.  
- **Feature Alignment:** Used `.align(join='left')` to ensure test dataset matched training features, filling missing categories with 0.  
- **Imputation:** Filled missing numerical/categorical values using median strategy (`SimpleImputer`).  
- **Feature Scaling:** Standardized features to mean 0, standard deviation 1 (`StandardScaler`).

---

### 🧠 2. Modeling Strategy

Several base models were cross-validated, including:

- Linear Regression  
- Decision Tree Regressor  
- Support Vector Regression  

The final pipeline focused on two **advanced tree-based models** and an ensemble:

- **Random Forest Regressor:**  
  - 500 estimators, `max_depth=20`, `min_samples_leaf=5`  
  - Acts as a stable, low-variance baseline

- **XGBoost Regressor:**  
  - `max_depth=2`, `reg_lambda=40`, `learning_rate=0.05`  
  - Regularized to capture complex non-linear patterns without overfitting

- **Hybrid Ensemble (Voting Regressor):**  
  - Combines Random Forest + XGBoost  
  - Weights: `Random Forest=1`, `XGBoost=2`  
  - Ensures stable, accurate predictions for unseen data

---

### 📊 3. Evaluation & Results

Models were evaluated using **Root Mean Squared Error (RMSE)** on log-transformed prices and **Residual Density Analysis**.

| Model       | RMSE (Log Scale) |
|------------|------------------|
| Random Forest | ~0.1526 |
| XGBoost      | ~0.1318 |
| Ensemble     | ~0.1350 |

---

### 🏆 Final Model Selection

Although **XGBoost achieved the lowest RMSE individually**, the **Ensemble Model** was selected as the final production model.  

Residual density analysis showed that the ensemble produced a **higher concentration of near-zero errors**, indicating **more stable and reliable predictions** on unseen data.

---

## 📈 Visualizations

*(Add your saved images from notebook here for better presentation)*

- **Actual vs Predicted Prices:** `images/actual_vs_predicted.png`  
- **Residual Distribution:** `images/residual_distribution.png`  
- **Model Comparison:** `images/model_comparison.png`  

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Kaggle-House-Prices-Prediction-Using-Sklearn.git

Install dependencies:

pip install -r requirements.txt

or manually:

pip install numpy pandas seaborn matplotlib scikit-learn xgboost tensorflow

Open and run the notebook:

notebooks/house_price_model.ipynb

The notebook will:

download the Kaggle dataset

train the models

generate submission.csv
---
## 🛠 Technologies Used

Python 3.x

NumPy, Pandas

Scikit-learn

XGBoost

TensorFlow (for reproducibility)

Matplotlib, Seaborn
---
