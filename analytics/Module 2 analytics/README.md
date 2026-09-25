
# Titanic Analytics Pipeline

## Project Overview

This project performs exploratory data analysis, classification modeling,
class imbalance comparison, Random Forest hyperparameter tuning, and
multivariate linear regression using the Titanic dataset.

## Dataset

The Titanic dataset was loaded using Seaborn and saved as `titanic.csv`
for offline use.

The cleaned dataset was used for EDA and modeling.

## Missing Value Handling

- Columns with less than 5% missing values: affected rows were removed.
- Age had between 5% and 30% missing values and was imputed using the median.
- Deck had a high percentage of missing values and missing values were
  represented using the `missing` category.

## EDA Findings

Age and fare distributions were explored using histograms and boxplots.
IQR-based outlier counts were calculated.

- Age outlier count: 65
- Fare outlier count: 114
- Fare mean: 32.097
- Fare median: 14.454
- Fare mode: 8.050
- Fare skewness: 4.801

Fare is positively skewed or right-skewed because the mean is greater than the median.

## Bivariate Analysis

Survival rates were calculated by sex, passenger class, and the combination
of sex and passenger class. Boolean masking using AND and OR conditions
was also demonstrated.

## Correlation Analysis

The correlation matrix used exactly these six columns:

- survived
- pclass
- age
- sibsp
- parch
- fare

The two strongest correlation pairs were:

- pclass and fare: -0.548
- sibsp and parch: 0.415

Correlation indicates association and does not establish causation.

## Multivariate Data Story

The charts show that survival rates differ by sex and passenger class.
The combined sex and passenger-class chart provides more detail than
examining either variable separately.

The age and fare charts show differences between survivors and
non-survivors, although the distributions overlap.

These observations describe associations in the Titanic dataset and
should not be interpreted as proof of causation.

## Standardization

Age and fare were standardized using the z-score formula during EDA.
This standardization was separate from the modeling preprocessing pipeline.

## Modeling

A stratified train/test split was used to preserve approximately similar
class proportions in the training and testing datasets.

The preprocessing pipeline used:

- Median imputation for numeric features
- Most-frequent imputation for categorical features
- One-hot encoding for categorical features
- StandardScaler for numeric features

All modeling preprocessing was fitted using training data through a
scikit-learn pipeline.

## Classification Results

| Model               |   Accuracy |   Precision |   Recall |   F1 Score |   AUC |
|:--------------------|-----------:|------------:|---------:|-----------:|------:|
| Logistic Regression |      0.809 |       0.783 |    0.691 |      0.734 | 0.861 |
| Decision Tree       |      0.809 |       0.815 |    0.647 |      0.721 | 0.856 |
| Random Forest       |      0.82  |       0.781 |    0.735 |      0.758 | 0.821 |

## Class Imbalance Results

| Strategy              |   Precision |   Recall |   F1 Score |
|:----------------------|------------:|---------:|-----------:|
| Baseline              |       0.783 |    0.691 |      0.734 |
| Class Weight Balanced |       0.718 |    0.75  |      0.734 |
| SMOTE                 |       0.735 |    0.735 |      0.735 |

The imbalance strategies were compared using precision, recall, and F1
score. The best imbalance strategy by F1 score was:

**SMOTE**

## Random Forest Grid Search

Best parameters:

```text
{'classifier__max_depth': None, 'classifier__max_features': 'sqrt', 'classifier__n_estimators': 100}
```

Best cross-validation F1 score:

**0.744**

Random Forest OOB score:

**0.795**

## Regression Results

| Model             |    MAE |   RMSE |    R2 |   Adjusted R2 |
|:------------------|-------:|-------:|------:|--------------:|
| Linear Regression | 21.099 | 41.702 | 0.348 |         0.309 |

The regression model predicted fare using other available passenger
features.

The residual spread is larger for higher predicted fares, which suggests possible heteroscedasticity.

Regression metrics are not directly comparable with classification
metrics because they measure a different prediction task.

## Final Recommendation

The initial classifier with the highest test-set F1 score was:

**Random Forest**

Its test metrics were:

- Accuracy: 0.820
- Precision: 0.781
- Recall: 0.735
- F1 score: 0.758
- ROC-AUC: 0.821

The F1 score was used to compare the balance between precision and recall.
The tuned Random Forest was also evaluated after hyperparameter tuning.

## Saved Files

- `titanic.csv`
- `outputs/age_histogram.png`
- `outputs/age_boxplot.png`
- `outputs/fare_histogram.png`
- `outputs/fare_boxplot.png`
- `outputs/correlation_heatmap.png`
- `outputs/survival_by_sex.png`
- `outputs/survival_by_pclass.png`
- `outputs/survival_by_sex_pclass.png`
- `outputs/age_by_survival.png`
- `outputs/fare_by_survival.png`
- `outputs/decision_tree.png`
- `outputs/confusion_matrices.png`
- `outputs/roc_curves.png`
- `outputs/residual_plot.png`
- `models/titanic_best_pipeline.joblib`

The saved pipeline contains preprocessing and the final estimator.
It was reloaded and tested on raw, unprocessed input data.
