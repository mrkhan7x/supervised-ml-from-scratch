# ML By Andrew NG Course 1 Project 

This document serves as your ultimate theoretical reference and checklist for Course 1 (Supervised Machine Learning: Regression and Classification). It maps out the mathematics, first-principles logic, and system configurations required to complete the capstone project.

---

## 📖 SECTION 1: Week-by-Week Mathematical Foundations

```
                            COURSE 1 MATHEMATICAL LANDSCAPE
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
  Linear Regression (Continuous)                                    Logistic Regression (Binary)
  ├─ Hypothesis: f(x) = w.x + b                                     ├─ Hypothesis: f(x) = g(w.x + b)
  ├─ Loss: Mean Squared Error (MSE)                                 ├─ Loss: Binary Cross-Entropy (Log-Loss)
  └─ Regularization: + (lambda/2m)*sum(w^2)                         └─ Regularization: + (lambda/2m)*sum(w^2)
```

---

### 1. Week 1: Univariate Linear Regression & Gradient Descent

#### Hypothesis Function
Predicts a continuous output $y$ from a single feature $x$:
$$f_{w,b}(x) = wx + b$$
*   **$w$ (Weight):** Slope of the line (determines the scale/impact of $x$).
*   **$b$ (Bias):** Y-intercept (determines the baseline prediction when $x=0$).

#### Cost Function: Mean Squared Error (MSE)
Measures the average squared difference between predictions and actual targets:
$$J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)^2$$
*   **Why $\frac{1}{2m}$?** The $2$ in the denominator is a mathematical convenience that cancels out when calculating the derivative (cost gradient) during updates.

#### Gradient Descent Optimization
Iteratively updates parameters to minimize cost $J(w,b)$:
$$w := w - \alpha \frac{\partial J(w,b)}{\partial w}$$
$$b := b - \alpha \frac{\partial J(w,b)}{\partial b}$$

Calculating partial derivatives yields:
$$\frac{\partial J(w,b)}{\partial w} = \frac{1}{m} \sum_{i=1}^m \left( f_{w,b}(x^{(i)}) - y^{(i)} \right) \cdot x^{(i)}$$
$$\frac{\partial J(w,b)}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left( f_{w,b}(x^{(i)}) - y^{(i)} \right)$$

*   **$\alpha$ (Learning Rate):** Adjusts step size.
    *   *Too small:* Training is slow (takes too many iterations).
    *   *Too large:* Diverges (overshoots minimum, causing cost to increase).

---

### 2. Week 2: Multivariate Linear Regression & Preprocessing

#### Hypothesis Function (Vectorized)
Handles multiple features $\vec{x} = [x_1, x_2, \dots, x_n]^T$:
$$f_{\vec{w},b}(\vec{x}) = \vec{w} \cdot \vec{x} + b = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$

In matrix form across all $m$ examples:
$$\vec{f}_{\vec{w},b}(\mathbf{X}) = \mathbf{X}\vec{w} + b$$
*   **$\mathbf{X}$:** Input matrix of shape $(m, n)$.
*   **$\vec{w}$:** Weights vector of shape $(n, 1)$ or $(n,)$.
*   **$b$:** Scalar bias added to each product element-wise.

#### Feature Scaling (Z-score Normalization)
Gradient descent converges slowly if features have wildly different ranges (e.g., $x_1 \in [1, 5]$ bedrooms, $x_2 \in [500, 5000]$ sqft) because the cost contours become highly elongated ellipses.
Standardization scales features to have a mean of $0$ and a standard deviation of $1$:

$$x_j^{(i)} := \frac{x_j^{(i)} - \mu_j}{\sigma_j}$$

*   **$\mu_j$ (Mean):** $\mu_j = \frac{1}{m}\sum_{i=1}^m x_j^{(i)}$
*   **$\sigma_j$ (Standard Deviation):** $\sigma_j = \sqrt{\frac{1}{m}\sum_{i=1}^m (x_j^{(i)} - \mu_j)^2}$

#### The Normal Equation (Analytical Alternative)
An analytical method to find optimal weights directly without iterations:
$$\vec{\theta} = \left(\mathbf{X}^T \mathbf{X}\right)^{-1} \mathbf{X}^T \vec{y}$$
*   *Note: In classic Stanford notation, $\mathbf{X}$ contains a dummy column of 1s representing $x_0 = 1$ to combine bias into $\vec{\theta}$.*
*   *Limitation:* Inverting $\mathbf{X}^T \mathbf{X}$ has a computational complexity of $\mathcal{O}(n^3)$. It becomes extremely slow when $n > 10,000$ features.

---

### 3. Week 3: Logistic Regression (Classification)

#### Hypothesis Function & Sigmoid Activation
Predicts probability of binary classes ($y \in \{0, 1\}$):
$$f_{\vec{w},b}(\vec{x}) = g(\vec{w} \cdot \vec{x} + b)$$

Where $g(z)$ is the Sigmoid activation function:
$$g(z) = \frac{1}{1 + e^{-z}}$$

#### Probability Output & Boundary Logic
*   **Output interpretation:** $f_{\vec{w},b}(\vec{x}) = P(y=1 \mid \vec{x}; \vec{w}, b)$
*   **Decision Boundary:** The contour where $f_{\vec{w},b}(\vec{x}) = 0.5$, which happens when $\vec{w} \cdot \vec{x} + b = 0$.
    *   $\vec{w} \cdot \vec{x} + b \ge 0 \implies$ Predict $y=1$
    *   $\vec{w} \cdot \vec{x} + b < 0 \implies$ Predict $y=0$

#### Cost Function: Binary Cross-Entropy (Log-Loss)
If we use Squared Error on Sigmoid outputs, the cost function is non-convex. To guarantee convexity, we use Log-Loss:
$$J(\vec{w},b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(f_{\vec{w},b}(\vec{x}^{(i)})) + (1 - y^{(i)})\log(1 - f_{\vec{w},b}(\vec{x}^{(i)})) \right]$$

---

### 4. Overfitting & Regularization

```
   Overfit (High Variance)           Regularized (Just Right)
        ▲                                    ▲
        │  * *                               │   * *
      * │ /   \   *                        * │  /   \ *
     ───┴─┴───┴───┴──►                     ───┴─┴───┴─┴──►
     (Wiggly, large weights)               (Smooth, shrunken weights)
```

#### L2 Regularization (Ridge Penalty)
Prevents overfitting by penalizing large weight magnitudes. We add a penalty term to the cost:
$$J(\vec{w},b)_{\text{reg}} = J(\vec{w},b) + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$
*   **$\lambda$ (Lambda):** Controls the penalty strength. 
    *   *Note: In Scikit-Learn, this is configured via parameter $C = \frac{1}{\lambda}$. Smaller $C$ means stronger regularization.*

#### Regularized Gradient Update (Weight Decay)
$$\frac{\partial J(\vec{w},b)_{\text{reg}}}{\partial w_j} = \left( \text{Original Derivative} \right) + \frac{\lambda}{m}w_j$$

Applying this to the gradient update step:
$$w_j := w_j\left(1 - \alpha \frac{\lambda}{m}\right) - \alpha \left( \text{Original Derivative} \right)$$
*   **Weight Shrinkage:** The multiplier $\left(1 - \alpha \frac{\lambda}{m}\right)$ decays the weights slightly on each step, preventing them from exploding to fit noise.

---

## 📊 SECTION 2: Advanced Diagnostic Metrics

### 1. Regression Metrics
*   **Mean Squared Error (MSE):** Measures variance:
    $$\text{MSE} = \frac{1}{m} \sum_{i=1}^m \left( y^{(i)} - \hat{y}^{(i)} \right)^2$$
*   **Root Mean Squared Error (RMSE):** Interpretable in target units:
    $$\text{RMSE} = \sqrt{\text{MSE}}$$
*   **Coefficient of Determination ($R^2$):** Measures the proportion of variance explained by features:
    $$R^2 = 1 - \frac{\sum (y^{(i)} - \hat{y}^{(i)})^2}{\sum (y^{(i)} - \bar{y})^2}$$
    *   *Interpretation:* $1.0$ is perfect; $0.0$ means your model performs no better than predicting the mean value $\bar{y}$.

### 2. Classification Metrics (Confusion Matrix)
```
                       True Class (y)
                     y = 1         y = 0
                 ┌────────────┬────────────┐
       y-hat = 1 │     TP     │     FP     │
Predicted        ├────────────┼────────────┤
       y-hat = 0 │     FN     │     TN     │
                 └────────────┴────────────┘
```
*   **Precision:** Accuracy of positive calls:
    $$\text{Precision} = \frac{TP}{TP + FP}$$
*   **Recall:** Coverage of positive samples:
    $$\text{Recall} = \frac{TP}{TP + FN}$$
*   **F1-Score:** Harmonic mean balancing Precision and Recall:
    $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

---

## 🏗️ SECTION 3: Capstone Project Specifications

You will build a **Housing Price & Quality Predictor** that implements both regression (predicting continuous price) and classification (predicting whether a house is a "Premium Deal" ($1$) or "Standard Deal" ($0$)) on the same dataset.

### Project Directory Layout
Create this layout in your workspace:
```
study_notes/Course 1/capstone/
   ├── README.md             <── (This file)
   ├── capstone_runner.py    <── (The runner template to code)
   └── app.py                <── (FastAPI deployment server)
```

### The Pipeline Tasks:
1.  **Scratch Math Engine:** Write vectorized NumPy code to standardize features, compute regularized regression gradients, compute regularized classification gradients, and calculate metrics.
2.  **Scikit-Learn Verification:** Build pipeline compositions using `PolynomialFeatures`, `StandardScaler`, and `Linear/LogisticRegression` estimators. Verify your scratch values match Scikit-Learn outputs.
3.  **Threshold Search:** Implement a loop searching for the optimal classification boundary threshold that maximizes the F1-Score.
4.  **Serving via FastAPI:** Serialize your models using `joblib` and build an API server exposing `/predict_price` and `/predict_deal` endpoints.
