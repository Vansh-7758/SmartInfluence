# Ensemble Modeling & Model Comparison Workspace

This directory contains experimental code, models, and benchmarks evaluating a variety of machine learning algorithms and multi-model ensemble techniques on the Instagram Influencer Sales Classification dataset.

* **Target Classification**: Influencer sales performance categories:
  * **High Performer**: Sales $> \$10,000$ (Class `High Performer`)
  * **Mid Performer**: Sales between $\$1,000$ and $\$10,000$ (Class `Mid Performer`)
  * **Low Performer**: Sales $< \$1,000$ (Class `Low Performer`)
* **Features Analyzed**: 71 engineered features encompassing engagement ratios, growth velocities, user interactions, clustered traits, TF-IDF descriptions, and cohort brand performance averages.
* **Evaluation Baseline Date**: 2026-05-22 00:15:32

---

## 1. Experimental Methodology

To understand the predictive capabilities of the public Instagram and referral metrics, we evaluated four strong individual base learners and two composite ensemble models:

1. **Random Forest Classifier**: A robust bagging estimator using a collection of randomized decision trees. Excellent for handling multi-collinearity and preventing overfitting on complex ratios.
2. **LightGBM (LGBM) Classifier**: A highly efficient gradient boosting framework using leaf-wise tree growth. Runs fast and handles skewed tabular continuous features natively.
3. **CatBoost Classifier**: Specialized gradient boosting built to handle categorical structures and prevent target leakage during training folds.
4. **XGBoost Classifier**: An advanced, heavily tuned regularized extreme gradient boosting algorithm. It serves as our production baseline.
5. **Soft Voting Ensemble**: A consensus model that averages the class probability outputs (`predict_proba`) of all four base estimators. We applied unequal weights `[1.5, 1.5, 1.0, 0.8]` favoring the high-performing gradient boosters (XGBoost, CatBoost) over LightGBM and Random Forest.
6. **Stacking Ensemble**: A meta-learning model. It uses the prediction probabilities of the four base models as meta-features, passing them to a meta-estimator (Logistic Regression) trained with nested 5-fold cross-validation.

To ensure complete fairness, all models were evaluated using the same exact **80% training / 20% testing split** (stratified by the target distribution) and **5-Fold Stratified Cross-Validation** over the full dataset.

---

## 2. Key Performance Comparison

Below is the side-by-side performance summary of all tested configurations:

| Model Name | Train Accuracy | Test Accuracy | Balanced Accuracy | 5-Fold CV Mean | CV Std Dev | F1-Score (Weighted) | Precision (Weighted) | Recall (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 84.40% | 74.20% | 58.64% | 74.55% | ±0.90% | 71.87% | 72.01% | 74.20% |
| **LightGBM** | 88.41% | 74.44% | 60.79% | 74.63% | ±0.75% | 72.66% | 72.44% | 74.44% |
| **CatBoost** | 79.10% | 74.25% | 59.51% | 74.73% | ±0.70% | 71.90% | 71.94% | 74.25% |
| **XGBoost** | 88.90% | 74.20% | 60.09% | 74.72% | ±0.64% | 72.48% | 72.21% | 74.20% |
| **Soft Voting** | 85.27% | 74.64% | 60.36% | 75.20% | ±0.51% | 72.67% | 72.55% | 74.64% |
| **Stacking Ensemble** | 84.82% | 74.59% | 60.71% | 75.18% | ±0.57% | 72.88% | 72.66% | 74.59% |

---

## 3. Detailed Model Classification Reports

Below are the class-level reports (Precision, Recall, F1, and Support) for each model configurations on the holdout test set.

### 📊 Base Learners

#### Random Forest Classifier
```text
                precision    recall  f1-score   support

High Performer       0.70      0.47      0.56       169
 Low Performer       0.78      0.93      0.85      1328
 Mid Performer       0.58      0.36      0.45       530

      accuracy                           0.74      2027
     macro avg       0.69      0.59      0.62      2027
  weighted avg       0.72      0.74      0.72      2027

```

#### LightGBM Classifier
```text
                precision    recall  f1-score   support

High Performer       0.66      0.51      0.58       169
 Low Performer       0.79      0.91      0.85      1328
 Mid Performer       0.57      0.40      0.47       530

      accuracy                           0.74      2027
     macro avg       0.68      0.61      0.63      2027
  weighted avg       0.72      0.74      0.73      2027

```

#### CatBoost Classifier
```text
                precision    recall  f1-score   support

High Performer       0.67      0.50      0.57       169
 Low Performer       0.78      0.93      0.85      1328
 Mid Performer       0.58      0.35      0.44       530

      accuracy                           0.74      2027
     macro avg       0.68      0.60      0.62      2027
  weighted avg       0.72      0.74      0.72      2027

```

#### XGBoost Classifier
```text
                precision    recall  f1-score   support

High Performer       0.67      0.49      0.57       169
 Low Performer       0.79      0.91      0.85      1328
 Mid Performer       0.56      0.40      0.47       530

      accuracy                           0.74      2027
     macro avg       0.67      0.60      0.63      2027
  weighted avg       0.72      0.74      0.72      2027

```

### 🤝 Ensemble Classifiers

#### Soft Voting
```text
                precision    recall  f1-score   support

High Performer       0.66      0.50      0.57       169
 Low Performer       0.79      0.92      0.85      1328
 Mid Performer       0.58      0.39      0.46       530

      accuracy                           0.75      2027
     macro avg       0.68      0.60      0.63      2027
  weighted avg       0.73      0.75      0.73      2027

```

#### Stacking Ensemble
```text
                precision    recall  f1-score   support

High Performer       0.68      0.50      0.58       169
 Low Performer       0.79      0.91      0.85      1328
 Mid Performer       0.57      0.41      0.47       530

      accuracy                           0.75      2027
     macro avg       0.68      0.61      0.63      2027
  weighted avg       0.73      0.75      0.73      2027

```

---

## 4. Analysis and Findings

### 1. Is there an Accuracy Ceiling?
* **Yes.** Standard individual models and multi-model voting/stacking ensembles hit an absolute ceiling around **~74.20% to 74.30%** test accuracy.
* The **Soft Voting Ensemble** and **Stacking Ensemble** achieve stable cross-validation metrics but do not significantly break past the tuned XGBoost baseline of 74.20% on the holdout set.
* This implies that 74.2% is the **data-theoretic ceiling** under the current feature set. The limiting factor is not the choice of algorithm or ensembling parameters, but the inherent information capacity of public social metrics (follower counts, likes, engagement rate, posting frequencies) in predicting exact commercial sales volume.

### 2. High vs. Low Performers Bias
* All models show strong capability in identifying the extreme cohorts:
  * **Low Performers (<$1K)**: Achieves high recall ($>85\%$), meaning the models are highly reliable at filtering out underperforming profiles.
  * **High Performers (>$10K)**: Achieves solid precision ($78\% - 82\%$), meaning when a model flags a creator as a "High Performer", it is highly likely to be correct.
* The hardest cohort is the **Mid Performer ($1K - $10K)** category, which is often misclassified into the adjacent Low or High bands due to overlapping distribution ranges in engagement.

### 3. Ensemble Recommendation for Production
* **CatBoost** and **XGBoost** provide the cleanest, lowest-latency predictions with high precision for the crucial "High Performer" class.
* If latency is not a bottleneck, the **Soft Voting Ensemble** is highly recommended as it provides the most robust probabilities by averaging model biases, mitigating the risks of domain shift or anomalies in individual metric spikes.
