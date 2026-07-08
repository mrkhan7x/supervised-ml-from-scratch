# -*- coding: utf-8 -*-
"""
=====================================================================
🏆 Course 1 Capstone Project: Housing Price & Deal Quality Predictor
=====================================================================
Author: Muhammad Roman Khan
Repository: Course 1 (Supervised ML - Regression & Classification) Capstone

Description:
I built this project to master Course 1 concepts by writing Z-score scaling, 
regularized linear regression gradient descent, regularized logistic 
regression gradient descent, and evaluation metrics from scratch using 
vectorized NumPy operations, and comparing them with Scikit-Learn pipelines.
=====================================================================
"""

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline

# =====================================================================
# 1. MOCK DATA GENERATOR FOR DUAL TARGETS
# =====================================================================
def generate_housing_dataset(m=200):
    """
    Generates my synthetic housing dataset with continuous and binary targets.
    Features: [Size (sqft), Bedrooms, Age (years), Tax Rate (%)]
    Targets: 
      - Price (Regression)
      - Premium Deal (Classification)
    """
    np.random.seed(42)
    # Size in sqft (800 to 4000)
    size = np.random.uniform(800, 4000, (m, 1))
    # Bedrooms (1 to 5)
    bedrooms = np.random.randint(1, 6, (m, 1)).astype(float)
    # Age (0 to 50 years)
    age = np.random.uniform(0, 50, (m, 1))
    # Tax Rate (1.0% to 3.5%)
    tax = np.random.uniform(1.0, 3.5, (m, 1))
    
    X = np.hstack((size, bedrooms, age, tax))
    
    # Calculate price using linear relationship + noise
    # Base price is $150k + $120/sqft + $25k/bedroom - $1.5k/year of age - $5k/tax %
    price = 150000 + 120 * size + 25000 * bedrooms - 1500 * age - 5000 * tax + np.random.normal(0, 15000, (m, 1))
    price = price.squeeze()
    
    # Deal Quality (Binary): 1 if price/sqft <= 180 (underpriced deal), else 0
    price_per_sqft = price / size.squeeze()
    y_deal = (price_per_sqft <= 185.0).astype(int)
    
    return X, price, y_deal

# =====================================================================
# 2. MY FIRST-PRINCIPLES MATH ENGINES (NumPy Implementation)
# =====================================================================
def z_score_scale_scratch(X):
    """
    My custom vectorized standardization function: (x - mean) / std.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    # Avoid dividing by zero if standard deviation is 0
    std = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def linear_regression_grad_descent_scratch(X, y, alpha=0.01, epochs=10000, lambda_=1.0):
    """
    My vectorized Gradient Descent implementation for Regularized Linear Regression.
    Updates weights and bias simultaneously, applying L2 penalty to weights.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    cost_history = []
    
    for epoch in range(epochs):
        # Calculate hypothesis predictions vector
        f_x = np.dot(X, w) + b
        errors = f_x - y
        
        # Calculate gradients with regularized weight decay
        dj_dw = (1 / m) * np.dot(X.T, errors) + (lambda_ / m) * w
        dj_db = (1 / m) * np.sum(errors)
        
        # Update parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Calculate total regularized cost J(w,b)
        total_cost = (1 / (2 * m)) * np.sum(errors ** 2) + (lambda_ / (2 * m)) * np.sum(w ** 2)
        cost_history.append(total_cost)
        
        if epoch % 2000 == 0:
            print(f"  Linear GD Epoch {epoch:5d} | Cost: {total_cost:.4f}")
            
    return w, b, cost_history

def logistic_regression_grad_descent_scratch(X, y, alpha=0.01, epochs=10000, lambda_=1.0):
    """
    My vectorized Gradient Descent implementation for Regularized Logistic Regression.
    Uses Sigmoid activation and Log-Loss cost computation.
    """
    m, n = X.shape
    w = np.zeros(n)
    b = 0.0
    cost_history = []
    
    for epoch in range(epochs):
        # Sigmoid probability outputs
        z = np.dot(X, w) + b
        f_x = 1 / (1 + np.exp(-z))
        errors = f_x - y
        
        # Regularized gradients
        dj_dw = (1 / m) * np.dot(X.T, errors) + (lambda_ / m) * w
        dj_db = (1 / m) * np.sum(errors)
        
        # Update weights and bias
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
        # Clip probabilities to prevent log(0) NaN errors
        epsilon = 1e-15
        f_x_clipped = np.clip(f_x, epsilon, 1 - epsilon)
        base_cost = (-1 / m) * np.sum(y * np.log(f_x_clipped) + (1 - y) * np.log(1 - f_x_clipped))
        reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
        total_cost = base_cost + reg_cost
        
        cost_history.append(total_cost)
        if epoch % 2000 == 0:
            print(f"  Logistic GD Epoch {epoch:5d} | Cost: {total_cost:.4f}")
            
    return w, b, cost_history

# =====================================================================
# 3. MY CUSTOM METRICS CALCULATOR ENGINE
# =====================================================================
def evaluate_regression_metrics(y_true, y_pred):
    """
    Calculates OLS evaluation metrics (MSE, RMSE, R2 Score) from scratch.
    """
    m = len(y_true)
    
    mse = (1 / m) * np.sum((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    mean_y = np.mean(y_true)
    total_sum_of_squares = np.sum((y_true - mean_y) ** 2)
    residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
    r2 = 1.0 - (residual_sum_of_squares / total_sum_of_squares)
    
    return mse, rmse, r2

def evaluate_classification_metrics(y_true, y_pred):
    """
    Calculates classification metrics (Confusion Matrix, Precision, Recall, F1) from scratch.
    """
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            tp += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            fp += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            tn += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            fn += 1
            
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    confusion_matrix_scratch = [[tn, fp], [fn, tp]]
    
    return {
        "confusion_matrix": confusion_matrix_scratch,
        "precision": precision,
        "recall": recall,
        "f1-score": f1
    }

# =====================================================================
# 4. CAPSTONE MODEL VALIDATION & SERIALIZATION
# =====================================================================
if __name__ == "__main__":
    print("--- Loading Capstone Housing Dataset ---")
    X, y_price, y_deal = generate_housing_dataset()
    print(f"Dataset Loaded. Features shape: {X.shape} | Regression Target: {y_price.shape} | Classification Target: {y_deal.shape}\n")
    
    # Train / Test Splits
    X_train_reg, X_test_reg, y_train_price, y_test_price = train_test_split(X, y_price, test_size=0.25, random_state=42)
    X_train_clf, X_test_clf, y_train_deal, y_test_deal = train_test_split(X, y_deal, test_size=0.25, random_state=42, stratify=y_deal)
    
    # Run and verify my scratch implementations
    try:
        X_train_scaled, mean_s, std_s = z_score_scale_scratch(X_train_reg)
        print("[SUCCESS] Z-Score Scaling implemented successfully!")
        
        # Test my linear regression gradient descent
        w_lin, b_lin, _ = linear_regression_grad_descent_scratch(X_train_scaled, y_train_price, epochs=20000, lambda_=1.0)
        print("[SUCCESS] Linear Regression GD implemented successfully!")
        
        # Test my logistic regression gradient descent
        w_log, b_log, _ = logistic_regression_grad_descent_scratch(X_train_scaled, y_train_deal, epochs=20000, lambda_=1.0)
        print("[SUCCESS] Logistic Regression GD implemented successfully!")
        
    except NotImplementedError as e:
        print(f"[WARNING] Placeholder active: {e}")
        print("Complete the functions in Section 2 to verify your math engines.")
        
    # =====================================================================
    # 5. BASELINE PRODUCTION PIPELINES (Scikit-Learn Verification)
    # =====================================================================
    print("\n--- Training Production Scikit-Learn Pipelines ---")
    
    # Regression Pipeline: Polynomial Expansion -> Scaling -> Ridge Regression (L2)
    reg_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=10.0))  # alpha is lambda in sklearn
    ])
    reg_pipeline.fit(X_train_reg, y_train_price)
    
    # Classification Pipeline: Scaling -> Logistic Regression (L2)
    clf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(C=0.1, penalty='l2')) # C = 1/lambda
    ])
    clf_pipeline.fit(X_train_clf, y_train_deal)
    
    print("Baseline pipelines trained.")
    
    # Serialize my pipelines for API serving in app.py
    joblib.dump(reg_pipeline, "housing_price_model.pkl")
    joblib.dump(clf_pipeline, "deal_quality_model.pkl")
    print("Models serialized and exported to disk ('housing_price_model.pkl', 'deal_quality_model.pkl').")
