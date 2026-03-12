# GPU Performance Prediction using Machine Learning

This project explores how different machine learning models perform when predicting GPU performance using hardware specifications.

The goal is to compare the effectiveness of:

- **Linear Regression** (baseline model)
- **Ridge Regression** (regularized linear model)
- **Random Forest Regression** (non-linear ensemble model)

The models are evaluated using **cross-validation** to understand how well they generalize to unseen data.

---

# Dataset

The dataset contains specifications for thousands of GPUs, including features such as:

- Memory Bandwidth
- Memory Speed
- Memory Bus Width
- ROPs (Raster Operations)
- Memory Type

These hardware features are used as predictors for the target performance metric.

---

# Modeling Approach

The project compares three regression models with increasing modeling complexity.

## 1. Linear Regression

Linear Regression serves as the **baseline model**.

It assumes a linear relationship between the GPU hardware features and the target variable.

$$
\hat{y} = w^T x + b
$$

Where:
- $x$ is the feature vector
- $w$ are the learned coefficients

Advantages:
- Simple
- Interpretable
- Fast to train

However, Linear Regression can struggle when features are **highly correlated**.

---

## 2. Ridge Regression

Ridge Regression extends Linear Regression by adding **L2 regularization**.

$$
\min_{\beta} \|y - X\beta\|^2 + \lambda \|\beta\|^2
$$

The regularization term penalizes large coefficients and helps reduce instability caused by correlated features.

Benefits:
- Improves model generalization
- Reduces overfitting
- Stabilizes coefficient estimates

---

## 3. Random Forest Regression

Random Forest is a **tree-based ensemble model** that can capture nonlinear relationships between features.

It works by training multiple decision trees and averaging their predictions.

$$
\hat{y} = \frac{1}{T}\sum_{t=1}^{T} f_t(x)
$$

Advantages:
- Captures nonlinear feature interactions
- Handles complex relationships
- Robust to noise

---

# Model Evaluation

Models are evaluated using **5-fold cross-validation**.

Metrics used:

### Mean Squared Error (MSE)

$$
MSE = \frac{1}{n}\sum (y - \hat{y})^2
$$

### $R^2$ Score

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

Higher $R^2$ and lower MSE indicate better model performance.

---

## Project Structure

```
gpu-regression-project/
├── All_GPUs.csv
├── gpu_regression.ipynb
├── README.md
└── requirements.txt
```

---

# Key Skills Demonstrated

- Machine Learning model comparison
- Feature engineering
- Cross-validation
- Regression modeling
- Ensemble methods

---

## Model Comparison Results

The models were evaluated using **5-fold cross-validation** with the following metrics:

- **$R^2$ Score** – measures how well the model explains the variance in the target variable (higher is better)
- **Mean Squared Error (MSE)** – measures prediction error (lower is better)

| Model | Mean $R^2$ | Std $R^2$ | Mean MSE | Std MSE |
|------|------|------|------|------|
| Random Forest | **0.9927** | 0.0014 | **0.0078** | 0.0020 |
| Ridge Linear Regression | 0.9223 | 0.0038 | 0.0826 | 0.0051 |
| Basic Linear Regression | 0.9223 | 0.0038 | 0.0826 | 0.0051 |

### Key Observations

- **Random Forest significantly outperforms the linear models**, achieving an $R^2$ score of **0.99**, indicating that it captures the relationships between GPU hardware features and the target variable extremely well.
- **Linear Regression and Ridge Regression perform similarly**, suggesting that regularization has minimal impact for this feature set.
- The strong performance of Random Forest indicates that **nonlinear relationships exist between GPU specifications and performance metrics**.

### Conclusion

Among the evaluated models, **Random Forest provides the best predictive performance**, demonstrating its ability to model complex interactions between GPU hardware features.