
from pathlib import Path
import warnings
import os
import joblib

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
ANALYTICS_DIR = PROJECT_ROOT
OUTPUT_DIR = ANALYTICS_DIR / "outputs"
MODEL_DIR = ANALYTICS_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = ANALYTICS_DIR / "titanic.csv"

print("=" * 70)
print("TITANIC ANALYTICS PIPELINE")
print("=" * 70)

print("Analytics directory:", ANALYTICS_DIR)
print("Output directory:", OUTPUT_DIR)
print("Model directory:", MODEL_DIR)


# ============================================================
# 2. LOAD DATASET ONCE AND SAVE OFFLINE FALLBACK
# ============================================================

print("\n" + "=" * 70)
print("DATASET LOADING")
print("=" * 70)

df = sns.load_dataset("titanic")

print("Dataset loaded successfully.")
print("Original shape:", df.shape)

# Immediately save the original dataset.
df.to_csv(CSV_PATH, index=False)

print("Original dataset saved to:", CSV_PATH)


# ============================================================
# 3. DATA PROFILING
# ============================================================

print("\n" + "=" * 70)
print("DATA PROFILING")
print("=" * 70)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nDataset shape:")
print(df.shape)

missing_report = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (
        df.isnull().mean() * 100
    ).round(2)
})

missing_report = missing_report[
    missing_report["Missing Count"] > 0
]

print("\nMissing value report:")
print(missing_report)


# ============================================================
# 4. MISSING VALUE HANDLING
# ============================================================

print("\n" + "=" * 70)
print("MISSING VALUE HANDLING")
print("=" * 70)

cleaned_df = df.copy()

print("\nMissing-value strategy:")

for column in cleaned_df.columns:
    missing_count = cleaned_df[column].isnull().sum()

    if missing_count > 0:
        percentage = (
            cleaned_df[column].isnull().mean() * 100
        )

        print(
            f"{column}: {percentage:.2f}% missing"
        )

# Under 5%: drop rows.
cleaned_df = cleaned_df.dropna(
    subset=["embarked", "embark_town"]
)

# Between 5% and 30%: median imputation.
cleaned_df["age"] = cleaned_df["age"].fillna(
    cleaned_df["age"].median()
)

# Above 30%: encode missing category.
cleaned_df["deck"] = (
    cleaned_df["deck"]
    .astype("object")
    .fillna("missing")
)

df = cleaned_df.copy()

print("\nCleaned dataset shape:", df.shape)
print("\nRemaining missing values:")
print(df.isnull().sum())

# Save cleaned CSV so it can also be used offline.
df.to_csv(CSV_PATH, index=False)

print("\nCleaned dataset saved to:", CSV_PATH)


# ============================================================
# 5. EDA: AGE AND FARE DISTRIBUTIONS
# ============================================================

print("\n" + "=" * 70)
print("EDA: AGE AND FARE")
print("=" * 70)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close()


# Age histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Distribution of Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
save_plot("age_histogram.png")


# Age boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["age"])
plt.title("Box Plot of Age")
plt.xlabel("Age")
save_plot("age_boxplot.png")


# Fare histogram
plt.figure(figsize=(8, 5))
sns.histplot(df["fare"], bins=30, kde=True)
plt.title("Distribution of Fare")
plt.xlabel("Fare")
plt.ylabel("Frequency")
save_plot("fare_histogram.png")


# Fare boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["fare"])
plt.title("Box Plot of Fare")
plt.xlabel("Fare")
save_plot("fare_boxplot.png")


# ============================================================
# 6. IQR OUTLIER ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("IQR OUTLIER ANALYSIS")
print("=" * 70)


def calculate_outliers(column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_bound)
        | (df[column] > upper_bound)
    ]

    print(f"\nColumn: {column}")
    print(f"Q1: {q1:.3f}")
    print(f"Q3: {q3:.3f}")
    print(f"IQR: {iqr:.3f}")
    print(f"Lower bound: {lower_bound:.3f}")
    print(f"Upper bound: {upper_bound:.3f}")
    print(f"Outlier count: {len(outliers)}")

    return len(outliers)


age_outlier_count = calculate_outliers("age")
fare_outlier_count = calculate_outliers("fare")


# ============================================================
# 7. FARE STATISTICS AND SKEWNESS
# ============================================================

print("\n" + "=" * 70)
print("FARE STATISTICS")
print("=" * 70)

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode().iloc[0]
fare_skewness = df["fare"].skew()

print(f"Mean: {fare_mean:.3f}")
print(f"Median: {fare_median:.3f}")
print(f"Mode: {fare_mode:.3f}")
print(f"Skewness coefficient: {fare_skewness:.3f}")

if fare_mean > fare_median:
    fare_skewness_conclusion = (
        "Fare is positively skewed or right-skewed "
        "because the mean is greater than the median."
    )
elif fare_mean < fare_median:
    fare_skewness_conclusion = (
        "Fare is negatively skewed or left-skewed "
        "because the mean is lower than the median."
    )
else:
    fare_skewness_conclusion = (
        "Mean and median are approximately equal."
    )

print(fare_skewness_conclusion)


# ============================================================
# 8. BIVARIATE SURVIVAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("BIVARIATE SURVIVAL ANALYSIS")
print("=" * 70)

survival_by_sex = (
    df.groupby("sex")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

survival_by_pclass = (
    df.groupby("pclass")["survived"]
    .mean()
    .mul(100)
    .round(2)
)

survival_by_sex_pclass = (
    df.groupby(["sex", "pclass"])["survived"]
    .mean()
    .mul(100)
    .round(2)
)

print("\nSurvival rate by sex:")
print(survival_by_sex)

print("\nSurvival rate by passenger class:")
print(survival_by_pclass)

print("\nSurvival rate by sex and passenger class:")
print(survival_by_sex_pclass)

# Boolean masking using AND (&)
female_first_class = df[
    (df["sex"] == "female")
    & (df["pclass"] == 1)
]

print(
    "\nFemale first-class survival rate:",
    round(
        female_first_class["survived"].mean() * 100,
        2
    ),
    "%"
)

# Boolean masking using OR (|)
first_or_second_class = df[
    (df["pclass"] == 1)
    | (df["pclass"] == 2)
]

print(
    "Class 1 OR Class 2 survival rate:",
    round(
        first_or_second_class["survived"].mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# 9. CORRELATION MATRIX AND HEATMAP
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION ANALYSIS")
print("=" * 70)

corr_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df[corr_columns].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    vmin=-1,
    vmax=1
)

plt.title("Correlation Matrix of Titanic Numeric Features")
save_plot("correlation_heatmap.png")


# Find strongest two off-diagonal correlations
abs_corr = corr_matrix.abs()

upper_triangle = abs_corr.where(
    np.triu(
        np.ones(abs_corr.shape),
        k=1
    ).astype(bool)
)

top_pairs = (
    upper_triangle
    .stack()
    .sort_values(ascending=False)
)

print("\nTwo strongest correlation pairs:")

top_correlation_text = []

for (feature1, feature2), abs_value in top_pairs.head(2).items():

    actual_value = corr_matrix.loc[
        feature1,
        feature2
    ]

    line = (
        f"{feature1} and {feature2}: "
        f"{actual_value:.3f}"
    )

    print(line)
    top_correlation_text.append(line)


# ============================================================
# 10. MULTIVARIATE CHARTS
# ============================================================

print("\n" + "=" * 70)
print("MULTIVARIATE VISUALIZATIONS")
print("=" * 70)


# Chart 1: Survival by sex
plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="sex",
    y="survived",
    estimator="mean"
)

plt.title("Survival Rate by Sex")
plt.xlabel("Sex")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

save_plot("survival_by_sex.png")


# Chart 2: Survival by passenger class
plt.figure(figsize=(7, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    estimator="mean"
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

save_plot("survival_by_pclass.png")


# Chart 3: Survival by sex and class
plt.figure(figsize=(8, 5))

sns.barplot(
    data=df,
    x="pclass",
    y="survived",
    hue="sex",
    estimator="mean"
)

plt.title("Survival Rate by Sex and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)

save_plot("survival_by_sex_pclass.png")


# Chart 4: Age and survival
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="age"
)

plt.title("Age Distribution by Survival Status")
plt.xlabel("Survival Status")
plt.ylabel("Age")

save_plot("age_by_survival.png")


# Chart 5: Fare and survival
plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="survived",
    y="fare"
)

plt.title("Fare Distribution by Survival Status")
plt.xlabel("Survival Status")
plt.ylabel("Fare")

save_plot("fare_by_survival.png")


# ============================================================
# 11. EDA STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("EDA STANDARDIZATION")
print("=" * 70)

standardization_columns = ["age", "fare"]

print("\nBefore standardization:")
print(df[standardization_columns].agg(["mean", "std"]))

df_standardized = df.copy()

for column in standardization_columns:

    mean_value = df[column].mean()
    std_value = df[column].std()

    df_standardized[column] = (
        df[column] - mean_value
    ) / std_value

print("\nAfter standardization:")
print(
    df_standardized[
        standardization_columns
    ].agg(["mean", "std"])
)


# ============================================================
# 12. MODELING FEATURES AND STRATIFIED SPLIT
# ============================================================

print("\n" + "=" * 70)
print("MODEL PREPARATION")
print("=" * 70)

model_features = [
    "age",
    "fare",
    "sibsp",
    "parch",
    "pclass",
    "sex",
    "embarked"
]

X = df[model_features].copy()
y = df["survived"].copy()

print("\nClass distribution:")
print(y.value_counts())

print("\nClass distribution percentage:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining class distribution:")
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("\nTesting class distribution:")
print(
    (y_test.value_counts(normalize=True) * 100).round(2)
)


# ============================================================
# 13. PREPROCESSING PIPELINE
# ============================================================

numeric_features = [
    "age",
    "fare",
    "sibsp",
    "parch",
    "pclass"
]

categorical_features = [
    "sex",
    "embarked"
]

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 14. TRAIN THREE CLASSIFIERS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING CLASSIFICATION MODELS")
print("=" * 70)

logistic_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

tree_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        DecisionTreeClassifier(
            random_state=42,
            max_depth=4
        )
    )
])

forest_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
    )
])

logistic_pipeline.fit(X_train, y_train)
tree_pipeline.fit(X_train, y_train)
forest_pipeline.fit(X_train, y_train)

print("Three classifiers trained successfully.")


# ============================================================
# 15. DECISION TREE VISUALIZATION
# ============================================================

tree_feature_names = (
    tree_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

decision_tree = (
    tree_pipeline
    .named_steps["classifier"]
)

plt.figure(figsize=(24, 12))

plot_tree(
    decision_tree,
    feature_names=tree_feature_names,
    class_names=[
        "Did not survive",
        "Survived"
    ],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree for Titanic Survival")

save_plot("decision_tree.png")


# ============================================================
# 16. EVALUATE THREE CLASSIFIERS
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION EVALUATION")
print("=" * 70)

models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": tree_pipeline,
    "Random Forest": forest_pipeline
}

classification_results = []
model_predictions = {}

for model_name, model in models.items():

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    classification_results.append({
        "Model": model_name,
        "TN": cm[0, 0],
        "FP": cm[0, 1],
        "FN": cm[1, 0],
        "TP": cm[1, 1],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc
    })

    model_predictions[model_name] = {
        "model": model,
        "predictions": predictions,
        "probabilities": probabilities
    }

results_df = pd.DataFrame(classification_results)

print("\nClassification comparison:")
print(results_df.round(3))


# Confusion matrices
fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 4)
)

for ax, (model_name, values) in zip(
    axes,
    model_predictions.items()
):

    cm = confusion_matrix(
        y_test,
        values["predictions"]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cbar=False,
        ax=ax
    )

    ax.set_title(model_name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrices.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ROC curves
plt.figure(figsize=(8, 6))

for model_name, values in model_predictions.items():

    probabilities = values["probabilities"]

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    auc_value = roc_auc_score(
        y_test,
        probabilities
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC={auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Titanic Classifiers")
plt.legend()
plt.grid(alpha=0.3)

save_plot("roc_curves.png")


# ============================================================
# 17. CLASS IMBALANCE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("CLASS IMBALANCE COMPARISON")
print("=" * 70)

baseline_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

weighted_model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )
    )
])

smote_model = ImbPipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "smote",
        SMOTE(random_state=42)
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

baseline_model.fit(X_train, y_train)
weighted_model.fit(X_train, y_train)
smote_model.fit(X_train, y_train)

imbalance_models = {
    "Baseline": baseline_model,
    "Class Weight Balanced": weighted_model,
    "SMOTE": smote_model
}

imbalance_results_list = []

for strategy, model in imbalance_models.items():

    predictions = model.predict(X_test)

    imbalance_results_list.append({
        "Strategy": strategy,
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y_test,
            predictions,
            zero_division=0
        )
    })

imbalance_results = pd.DataFrame(
    imbalance_results_list
)

print("\nImbalance comparison:")
print(imbalance_results.round(3))

best_imbalance_row = imbalance_results.loc[
    imbalance_results["F1 Score"].idxmax()
]

best_imbalance_strategy = (
    best_imbalance_row["Strategy"]
)

print(
    "\nHighest F1 imbalance strategy:",
    best_imbalance_strategy
)


# ============================================================
# 18. RANDOM FOREST GRID SEARCH
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST GRID SEARCH")
print("=" * 70)

rf_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "classifier",
        RandomForestClassifier(
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
    )
])

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__max_features": [
        "sqrt",
        "log2"
    ]
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    return_train_score=True
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

best_rf = (
    best_model
    .named_steps["classifier"]
)

print("\nBest parameters:")
print(grid_search.best_params_)

print(
    "\nBest cross-validation F1:",
    round(grid_search.best_score_, 3)
)

print(
    "\nRandom Forest OOB score:",
    round(best_rf.oob_score_, 3)
)

tuned_predictions = best_model.predict(X_test)

print(
    "\nTuned Random Forest test accuracy:",
    round(
        accuracy_score(
            y_test,
            tuned_predictions
        ),
        3
    )
)

print(
    "Tuned Random Forest test F1:",
    round(
        f1_score(
            y_test,
            tuned_predictions
        ),
        3
    )
)


# ============================================================
# 19. REGRESSION: PREDICT FARE
# ============================================================

print("\n" + "=" * 70)
print("FARE REGRESSION")
print("=" * 70)

regression_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "sex",
    "embarked"
]

X_reg = df[regression_features].copy()
y_reg = df["fare"].copy()

numeric_features_reg = X_reg.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features_reg = X_reg.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

X_reg_train, X_reg_test, y_reg_train, y_reg_test = (
    train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42
    )
)

numeric_transformer_reg = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])

categorical_transformer_reg = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

preprocessor_reg = ColumnTransformer([
    (
        "numeric",
        numeric_transformer_reg,
        numeric_features_reg
    ),
    (
        "categorical",
        categorical_transformer_reg,
        categorical_features_reg
    )
])

regression_pipeline = Pipeline([
    (
        "preprocessor",
        preprocessor_reg
    ),
    (
        "regressor",
        LinearRegression()
    )
])

regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)

y_reg_pred = regression_pipeline.predict(
    X_reg_test
)

mae = mean_absolute_error(
    y_reg_test,
    y_reg_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_pred
    )
)

r2 = r2_score(
    y_reg_test,
    y_reg_pred
)

regression_feature_names = (
    regression_pipeline
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

p = len(regression_feature_names)
n = len(y_reg_test)

if n > p + 1:

    adjusted_r2 = 1 - (
        ((1 - r2) * (n - 1))
        / (n - p - 1)
    )

else:

    adjusted_r2 = np.nan

print("\nRegression metrics:")
print("MAE:", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R2:", round(r2, 3))
print("Adjusted R2:", round(adjusted_r2, 3))


# Residual plot
residuals = y_reg_test - y_reg_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Plot - Fare Regression")
plt.grid(alpha=0.3)

save_plot("residual_plot.png")


# Basic residual interpretation
residual_spread_low = np.std(
    residuals[
        y_reg_pred <= np.median(y_reg_pred)
    ]
)

residual_spread_high = np.std(
    residuals[
        y_reg_pred > np.median(y_reg_pred)
    ]
)

if residual_spread_high > residual_spread_low * 1.25:

    residual_conclusion = (
        "The residual spread is larger for higher predicted fares, "
        "which suggests possible heteroscedasticity."
    )

else:

    residual_conclusion = (
        "The residual spread does not show a strong automatic "
        "indication of heteroscedasticity using this simple comparison. "
        "The residual plot should be reviewed visually."
    )

print("\nResidual interpretation:")
print(residual_conclusion)


# ============================================================
# 20. FINAL COMPARISON TABLES
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

final_classification = results_df[
    [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "AUC"
    ]
].copy()

final_classification = final_classification.round(3)

final_regression = pd.DataFrame({
    "Model": ["Linear Regression"],
    "MAE": [mae],
    "RMSE": [rmse],
    "R2": [r2],
    "Adjusted R2": [adjusted_r2]
})

final_regression = final_regression.round(3)

print("\nClassification metrics:")
print(final_classification)

print("\nRegression metrics:")
print(final_regression)

final_classification.to_csv(
    OUTPUT_DIR / "classification_results.csv",
    index=False
)

imbalance_results.to_csv(
    OUTPUT_DIR / "imbalance_results.csv",
    index=False
)

final_regression.to_csv(
    OUTPUT_DIR / "regression_results.csv",
    index=False
)


# ============================================================
# 21. FINAL MODEL RECOMMENDATION
# ============================================================

best_initial_model_row = results_df.loc[
    results_df["F1 Score"].idxmax()
]

best_initial_model_name = (
    best_initial_model_row["Model"]
)

best_initial_f1 = best_initial_model_row["F1 Score"]
best_initial_accuracy = best_initial_model_row["Accuracy"]
best_initial_precision = best_initial_model_row["Precision"]
best_initial_recall = best_initial_model_row["Recall"]
best_initial_auc = best_initial_model_row["AUC"]

print("\n" + "=" * 70)
print("FINAL MODEL RECOMMENDATION")
print("=" * 70)

print(
    f"Best initial classifier by F1 score: "
    f"{best_initial_model_name}"
)

print(
    f"Accuracy: {best_initial_accuracy:.3f}"
)

print(
    f"Precision: {best_initial_precision:.3f}"
)

print(
    f"Recall: {best_initial_recall:.3f}"
)

print(
    f"F1 Score: {best_initial_f1:.3f}"
)

print(
    f"ROC-AUC: {best_initial_auc:.3f}"
)

print(
    "\nThe tuned Random Forest was evaluated separately after "
    "GridSearchCV. Classification metrics and regression metrics "
    "represent different prediction tasks and should not be directly "
    "compared."
)


# ============================================================
# 22. SAVE AND RELOAD COMPLETE PIPELINE
# ============================================================

print("\n" + "=" * 70)
print("SAVE AND RELOAD MODEL")
print("=" * 70)

model_path = MODEL_DIR / "titanic_best_pipeline.joblib"

joblib.dump(
    best_model,
    model_path
)

print("Pipeline saved to:", model_path)
print("File exists:", model_path.exists())

loaded_pipeline = joblib.load(model_path)

print("Pipeline reloaded successfully.")

raw_test_samples = X_test.iloc[:10].copy()

original_predictions = best_model.predict(
    raw_test_samples
)

loaded_predictions = loaded_pipeline.predict(
    raw_test_samples
)

predictions_match = np.array_equal(
    original_predictions,
    loaded_predictions
)

print("\nOriginal predictions:")
print(original_predictions)

print("\nReloaded predictions:")
print(loaded_predictions)

print("\nPredictions match:", predictions_match)


# ============================================================
# 23. AUTOMATIC README GENERATION
# ============================================================

print("\n" + "=" * 70)
print("CREATING README")
print("=" * 70)

top_correlation_text_value = "\n".join(
    [
        f"- {value}"
        for value in top_correlation_text
    ]
)

classification_markdown = final_classification.to_markdown(
    index=False
)

regression_markdown = final_regression.to_markdown(
    index=False
)

imbalance_markdown = imbalance_results.round(3).to_markdown(
    index=False
)

readme_content = f"""
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

- Age outlier count: {age_outlier_count}
- Fare outlier count: {fare_outlier_count}
- Fare mean: {fare_mean:.3f}
- Fare median: {fare_median:.3f}
- Fare mode: {fare_mode:.3f}
- Fare skewness: {fare_skewness:.3f}

{fare_skewness_conclusion}

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

{top_correlation_text_value}

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

{classification_markdown}

## Class Imbalance Results

{imbalance_markdown}

The imbalance strategies were compared using precision, recall, and F1
score. The best imbalance strategy by F1 score was:

**{best_imbalance_strategy}**

## Random Forest Grid Search

Best parameters:

```text
{grid_search.best_params_}
```

Best cross-validation F1 score:

**{grid_search.best_score_:.3f}**

Random Forest OOB score:

**{best_rf.oob_score_:.3f}**

## Regression Results

{regression_markdown}

The regression model predicted fare using other available passenger
features.

{residual_conclusion}

Regression metrics are not directly comparable with classification
metrics because they measure a different prediction task.

## Final Recommendation

The initial classifier with the highest test-set F1 score was:

**{best_initial_model_name}**

Its test metrics were:

- Accuracy: {best_initial_accuracy:.3f}
- Precision: {best_initial_precision:.3f}
- Recall: {best_initial_recall:.3f}
- F1 score: {best_initial_f1:.3f}
- ROC-AUC: {best_initial_auc:.3f}

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
"""

readme_path = ANALYTICS_DIR / "README.md"

with open(readme_path, "w", encoding="utf-8") as file:
    file.write(readme_content)

print("README saved to:", readme_path)


# ============================================================
# 24. FINAL COMPLETION CHECK
# ============================================================

print("\n" + "=" * 70)
print("PROJECT COMPLETED")
print("=" * 70)

required_files = [
    CSV_PATH,
    OUTPUT_DIR / "age_histogram.png",
    OUTPUT_DIR / "age_boxplot.png",
    OUTPUT_DIR / "fare_histogram.png",
    OUTPUT_DIR / "fare_boxplot.png",
    OUTPUT_DIR / "correlation_heatmap.png",
    OUTPUT_DIR / "decision_tree.png",
    OUTPUT_DIR / "confusion_matrices.png",
    OUTPUT_DIR / "roc_curves.png",
    OUTPUT_DIR / "residual_plot.png",
    MODEL_DIR / "titanic_best_pipeline.joblib",
    ANALYTICS_DIR / "README.md"
]

print("\nRequired file check:")

for file_path in required_files:
    print(
        f"{file_path.name}: "
        f"{'FOUND' if file_path.exists() else 'MISSING'}"
    )

print("\nPipeline execution finished.")