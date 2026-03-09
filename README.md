# House Prices Prediction: Advanced Regression Techniques
---

## 📁 Folder Structure
Kaggle-House-Prices-Prediction-Using-Sklearn/
```
├── data/                     # Contains Kaggle train.csv and test.csv
│   ├── train.csv
│   └── test.csv
│
├── notebooks/                # Main training and analysis notebooks
│   └── house_price_model.ipynb
│
├── model/                   # Saved trained models
│   └── house_price_model.h5
│
├── submission/              # Generated submission files
│   └── submission.csv
│
├── README.md                 # Project overview and documentation
└── requirements.txt          # List of dependencies

```
---
## 📌 Project Overview
This repository contains a complete machine learning pipeline to predict housing prices based on the [Kaggle House Prices Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition. The project demonstrates advanced data preprocessing, model regularization, and the creation of a custom weighted ensemble model to achieve high-precision predictions.
---
## 📊 Dataset
The dataset includes 79 explanatory variables describing almost every aspect of residential homes in Ames, Iowa. 
* `train.csv`: Used for training and validation.
* `test.csv`: Used for generating final Kaggle leaderboard predictions.
---
## ⚙️ Machine Learning Pipeline

### 🗂 1. Data Preprocessing
To ensure the models receive clean, standardized data, the following steps were implemented:
* **Target Transformation:** Applied `np.log1p` to the `SalePrice` to handle right-skewness and normalize the distribution.
* **Categorical Encoding:** Utilized One-Hot Encoding (`pd.get_dummies`) for categorical features.
* **Feature Alignment:** Used `.align(join='left')` to ensure the test dataset perfectly matched the training dataset's feature space, filling missing categories with 0s to prevent data leakage and model crashes.
* **Imputation:** Handled missing numerical and encoded categorical values using a median strategy (`SimpleImputer`).
* **Feature Scaling:** Standardized all features to have a mean of 0 and standard deviation of 1 (`StandardScaler`).
---
### 🧠 2. Modeling Strategy
Several base models were cross-validated (Linear Regression, Decision Tree, SVM). The final architecture focused on tuning two advanced tree-based models and combining them:

* **Random Forest Regressor:** Built with 500 estimators and strict depth limits (`max_depth=20`, `min_samples_leaf=5`) to act as a highly stable, low-variance baseline.
* **XGBoost Regressor:** Heavily regularized (`max_depth=2`, `reg_lambda=40`, `learning_rate=0.05`) to prevent overfitting while capturing complex, non-linear relationships.
* **Hybrid Ensemble (The Champion):** A `VotingRegressor` combining the Random Forest and XGBoost models. XGBoost was given a higher weight (`weights=[1, 2]`) to prioritize its accuracy, while the Random Forest acted as a stabilizer against outliers.
---

### 📊 3. Evaluation & Results
Models were evaluated using **Root Mean Squared Error (RMSE)** on the log-transformed prices and through **Residual Density Analysis**.

* **Random Forest RMSE:** ~0.1526
* **XGBoost RMSE:** ~0.1318
* **Ensemble RMSE:** ~0.1350
---

**Conclusion:**  While XGBoost achieved a slightly lower absolute RMSE, the **Ensemble Model** was selected as the final production model. Residual analysis curves demonstrated that the Ensemble had the highest density of near-zero errors, making it the most reliable and consistent predictor for unseen data.
---

## 🚀 How to Run
1. Clone this repository.
2. Ensure you have your `kaggle.json` API key ready if running via Google Colab.
3. Install dependencies: `pip install numpy pandas seaborn matplotlib scikit-learn xgboost tensorflow`
4. Run the notebook to automatically download the data, train the ensemble, and generate the `submission.csv` file.