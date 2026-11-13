
# ============================================================
# TASK 6: SVM CLASSIFIER + KERNEL SHAP
# Stroke classification coursework
# ============================================================

import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import sklearn
import imblearn

from utils import DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="X does not have valid feature names"
)


# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------

RANDOM_STATE = 42

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Kernel SHAP settings
BACKGROUND_SIZE = 10
SHAP_NSAMPLES = 50

print("SHAP version:", shap.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Imbalanced-learn version:", imblearn.__version__)


# ------------------------------------------------------------
# 2. LOAD AND PREPROCESS DATA
# ------------------------------------------------------------

# Use EXACTLY the same DataLoader workflow as Random Forest.

data_loader = DataLoader()
data_loader.load_dataset()

# Split the data for evaluation
X_train, X_val, X_test, y_train, y_val, y_test = (
    data_loader.get_data_split()
)

# Preserve original training data before oversampling
X_train_original = X_train.copy()

# Preserve original training labels
y_train_original = y_train.copy()

# Oversample training data only
X_train, y_train = data_loader.oversample(
    X_train,
    y_train
)

print("\n========== DATASET SPLIT ==========")

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTraining class distribution after oversampling:")
print(y_train.value_counts())

print("\nOriginal training class distribution:")
print(y_train_original.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())

# Ensure feature columns are identical
feature_columns = X_train_original.columns.tolist()

X_val = X_val.reindex(
    columns=feature_columns,
    fill_value=0
)

X_test = X_test.reindex(
    columns=feature_columns,
    fill_value=0
)

assert list(X_train_original.columns) == list(X_val.columns)
assert list(X_train_original.columns) == list(X_test.columns)

# ------------------------------------------------------------
# 3. FEATURE SCALING
# ------------------------------------------------------------

# Fit the scaler using ORIGINAL training data only.
# This prevents duplicated oversampled observations
# from affecting the scaling parameters.

scaler = StandardScaler()

scaler.fit(X_train_original)

# Transform oversampled training data
X_train_scaled = scaler.transform(X_train)

# Transform validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("\n========== FEATURE SCALING ==========")
print("Scaler fitted using original training data.")
print("Training scaled shape:", X_train_scaled.shape)
print("Validation scaled shape:", X_val_scaled.shape)
print("Testing scaled shape:", X_test_scaled.shape)


# ------------------------------------------------------------
# 4. TRAIN SVM
# ------------------------------------------------------------

# RBF kernel allows nonlinear decision boundaries.
# probability=True enables predict_proba for Kernel SHAP.
# class_weight='balanced' accounts for class imbalance.

svm = SVC(
    kernel="rbf",
    C=0.05,
    gamma=0.01,
    probability=True,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

print("\n========== TRAINING SVM ==========")

start_time = time.time()

svm.fit(
    X_train_scaled,
    y_train
)

training_time = time.time() - start_time

print("SVM training completed.")
print(f"Training time: {training_time:.2f} seconds")


# ------------------------------------------------------------
# 5. VALIDATION THRESHOLD SELECTION
# ------------------------------------------------------------

# Obtain positive-class probabilities.
# Class 1 represents recorded stroke.

positive_class_index = list(svm.classes_).index(1)

val_probabilities = svm.predict_proba(
    X_val_scaled
)[:, positive_class_index]

threshold_results = []

for threshold in THRESHOLDS:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Balanced Accuracy": balanced_accuracy_score(
            y_val, val_predictions
        ),
        "Stroke Precision": precision_score(
            y_val, val_predictions, zero_division=0
        ),
        "Stroke Recall": recall_score(
            y_val, val_predictions, zero_division=0
        ),
        "Stroke F1": f1_score(
            y_val, val_predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

print("\n========== VALIDATION THRESHOLDS ==========")
print(threshold_df.to_string(index=False))

# Select the threshold with the highest balanced accuracy.
# Ties favour the lower threshold.

threshold_df = threshold_df.sort_values(
    by=["Balanced Accuracy", "Threshold"],
    ascending=[False, True]
)

selected_threshold = float(
    threshold_df.iloc[0]["Threshold"]
)

print("\nSelected threshold:", selected_threshold)

print(
    "Selection rule: Highest validation balanced accuracy "
    "among the tested thresholds."
)


# ------------------------------------------------------------
# 6. FINAL TEST EVALUATION
# ------------------------------------------------------------

test_probabilities = svm.predict_proba(
    X_test_scaled
)[:, positive_class_index]

test_predictions = (
    test_probabilities >= selected_threshold
).astype(int)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

test_balanced_accuracy = balanced_accuracy_score(
    y_test, test_predictions
)

test_precision = precision_score(
    y_test, test_predictions, zero_division=0
)

test_recall = recall_score(
    y_test, test_predictions, zero_division=0
)

test_f1 = f1_score(
    y_test, test_predictions, zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test, test_probabilities
)

test_average_precision = average_precision_score(
    y_test, test_probabilities
)

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

print("\n========== SVM TEST PERFORMANCE ==========")

print("Selected threshold:", selected_threshold)
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Balanced accuracy: {test_balanced_accuracy:.4f}")
print(f"Stroke precision: {test_precision:.4f}")
print(f"Stroke recall: {test_recall:.4f}")
print(f"Stroke F1-score: {test_f1:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(f"Average Precision: {test_average_precision:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=["No recorded stroke", "Recorded stroke"],
        zero_division=0
    )
)

tn, fp, fn, tp = cm.ravel()

print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# ------------------------------------------------------------
# 7. CONFUSION MATRIX PLOT
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 5))

im = ax.imshow(cm, interpolation="nearest")

ax.figure.colorbar(im, ax=ax)

ax.set(
    xticks=np.arange(2),
    yticks=np.arange(2),
    xticklabels=["No stroke", "Stroke"],
    yticklabels=["No stroke", "Stroke"],
    xlabel="Predicted class",
    ylabel="Actual class",
    title="SVM Confusion Matrix"
)

for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. FIND THE SAME LOCAL CASE USED IN TASK 3
# ------------------------------------------------------------

# Task 3 case:
# Actual stroke label = 1
# Age = 39
# Average glucose = 97.76

case_mask = (
    (X_test["age"] == 39) &
    np.isclose(
        X_test["avg_glucose_level"],
        97.76,
        atol=0.01
    ) &
    (y_test == 1)
)

case_positions = np.flatnonzero(
    case_mask.to_numpy()
)

if len(case_positions) == 0:
    raise ValueError(
        "The Task 3 case was not found. "
        "Check that the DataLoader split is identical "
        "to the Random Forest and Task 3 split."
    )

case_position = int(case_positions[0])

case_X = X_test.iloc[[case_position]]

case_actual = int(y_test.iloc[case_position])

case_probability = float(
    test_probabilities[case_position]
)

case_prediction = int(
    test_predictions[case_position]
)

print("\n========== SAME LOCAL CASE ==========")

print("Test position (zero-based):", case_position)
print("Actual class:", case_actual)
print("SVM predicted class:", case_prediction)
print("SVM Class 1 probability:", case_probability)
print("Selected threshold:", selected_threshold)

if case_prediction == 0 and case_actual == 1:
    print("SVM result: False negative")

elif case_prediction == 1 and case_actual == 1:
    print("SVM result: True positive")

else:
    print("SVM result: Different outcome under the SVM.")


# ------------------------------------------------------------
# 9. KERNEL SHAP EXPLAINER
# ------------------------------------------------------------

print("\n========== KERNEL SHAP ==========")

# Use original training observations as the background.
# Background is selected BEFORE oversampling.

background = shap.sample(
    X_train_original,
    nsamples=min(
        BACKGROUND_SIZE,
        len(X_train_original)
    ),
    random_state=RANDOM_STATE
)

background_array = np.asarray(
    background,
    dtype=float
)

# Prediction function:
# Input = original encoded, unscaled features.
# Apply the same fitted scaler before SVM prediction.
# Output = Class 1 predicted probability.

def predict_stroke_probability(data):

    data = np.asarray(data, dtype=float)

    # Restore feature names before scaling
    data_df = pd.DataFrame(
        data,
        columns=feature_columns
    )

    scaled_data = scaler.transform(data_df)

    return svm.predict_proba(
        scaled_data
    )[:, positive_class_index]


explainer = shap.KernelExplainer(
    predict_stroke_probability,
    background_array
)

# Explain ALL held-out test observations.
# Do not reduce this to a smaller subset.

X_shap = X_test.copy()

print("Explanation sample:", X_shap.shape)
print("Background sample:", background_array.shape)
print("Explained output: Class 1 predicted probability")
print("Kernel SHAP nsamples:", SHAP_NSAMPLES)

shap_start_time = time.time()

shap_values = explainer.shap_values(
    X_shap.to_numpy(dtype=float),
    nsamples=SHAP_NSAMPLES,
    l1_reg="num_features(21)",
    silent=True
)

shap_runtime = time.time() - shap_start_time

print(f"\nKernel SHAP runtime: {shap_runtime:.2f} seconds")

# Convert SHAP output to NumPy array.
# Prediction function returns a single output.

if isinstance(shap_values, list):
    stroke_shap_values = np.asarray(
        shap_values[0]
    )
else:
    stroke_shap_values = np.asarray(
        shap_values
    )

if stroke_shap_values.ndim == 3:
    stroke_shap_values = stroke_shap_values[:, :, 0]

base_value = float(
    np.asarray(
        explainer.expected_value
    ).reshape(-1)[0]
)

print("SHAP values shape:", stroke_shap_values.shape)
print("Base value:", base_value)

assert stroke_shap_values.shape == (
    len(X_shap),
    X_shap.shape[1]
)


# ------------------------------------------------------------
# 10. KERNEL SHAP ADDITIVITY / RECONSTRUCTION CHECK
# ------------------------------------------------------------

actual_probabilities = predict_stroke_probability(
    X_shap.to_numpy(dtype=float)
)

reconstructed_probabilities = (
    base_value +
    stroke_shap_values.sum(axis=1)
)

residuals = (
    actual_probabilities -
    reconstructed_probabilities
)

max_abs_residual = float(
    np.max(np.abs(residuals))
)

mean_abs_residual = float(
    np.mean(np.abs(residuals))
)

print("\n========== SHAP RECONSTRUCTION CHECK ==========")

print("Maximum absolute residual:", max_abs_residual)
print("Mean absolute residual:", mean_abs_residual)

print(
    "Maximum residual observation:",
    int(np.argmax(np.abs(residuals)))
)

verification_passed = np.allclose(
    actual_probabilities,
    reconstructed_probabilities,
    atol=1e-5,
    rtol=1e-5
)

# ------------------------------------------------------------
# 11. GLOBAL SHAP FEATURE IMPORTANCE
# ------------------------------------------------------------

mean_abs_shap = np.abs(
    stroke_shap_values
).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_shap.columns,
    "Mean Absolute SHAP": mean_abs_shap
}).sort_values(
    by="Mean Absolute SHAP",
    ascending=False
).reset_index(drop=True)

importance_df.insert(
    0,
    "Rank",
    np.arange(1, len(importance_df) + 1)
)

print("\n========== GLOBAL SVM SHAP IMPORTANCE ==========")

print(
    importance_df.to_string(index=False)
)


# ------------------------------------------------------------
# 12. GLOBAL SHAP SUMMARY BEESWARM PLOT
# ------------------------------------------------------------

print("\nDisplaying global SHAP beeswarm plot...")

shap.summary_plot(
    stroke_shap_values,
    X_shap,
    feature_names=X_shap.columns.tolist(),
    max_display=21,
    show=True
)


# ------------------------------------------------------------
# 13. GLOBAL SHAP BAR PLOT
# ------------------------------------------------------------

print("\nDisplaying global SHAP bar plot...")

shap.summary_plot(
    stroke_shap_values,
    X_shap,
    feature_names=X_shap.columns.tolist(),
    plot_type="bar",
    max_display=21,
    show=True
)


# ------------------------------------------------------------
# 14. LOCAL SHAP EXPLANATION FOR THE SAME TASK 3 CASE
# ------------------------------------------------------------

local_shap_values = stroke_shap_values[
    case_position
]

local_df = pd.DataFrame({
    "Feature": X_shap.columns,
    "Patient Value": X_shap.iloc[
        case_position
    ].values,
    "SHAP Value": local_shap_values
})

local_df["Absolute SHAP"] = (
    local_df["SHAP Value"].abs()
)

local_df = local_df.sort_values(
    by="Absolute SHAP",
    ascending=False
)

print("\n========== LOCAL SVM SHAP EXPLANATION ==========")

print(
    local_df.to_string(index=False)
)


# ------------------------------------------------------------
# 15. LOCAL WATERFALL PLOT
# ------------------------------------------------------------

local_explanation = shap.Explanation(
    values=local_shap_values,
    base_values=base_value,
    data=X_shap.iloc[
        case_position
    ].to_numpy(dtype=float),
    feature_names=X_shap.columns.tolist()
)

print("\nDisplaying local SHAP waterfall plot...")

shap.plots.waterfall(
    local_explanation,
    max_display=21,
    show=True
)


# ------------------------------------------------------------
# 16. SOFTWARE AND COMPUTATION SUMMARY
# ------------------------------------------------------------

print("\n========== TASK 6 SUMMARY ==========")

print("Model: RBF-kernel Support Vector Machine")
print("Explainer: shap.KernelExplainer")
print("Output: Class 1 predicted probability")

print("Background size:", len(background))
print("Background source: Original training data")
print("Background random seed:", RANDOM_STATE)

print("Explained observations:", len(X_shap))
print("Number of features:", X_shap.shape[1])

print("Kernel SHAP nsamples:", SHAP_NSAMPLES)

print(f"Training runtime: {training_time:.2f} seconds")
print(f"SHAP runtime: {shap_runtime:.2f} seconds")

print("Maximum absolute residual:", max_abs_residual)
print("Mean absolute residual:", mean_abs_residual)

print("Reconstruction verification:", verification_passed)

print("\n========== TASK 6 COMPLETED ==========")
