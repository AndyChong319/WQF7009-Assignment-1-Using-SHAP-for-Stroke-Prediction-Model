
# %% Imports

from utils import DataLoader

import sklearn
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score
)


# %% Load and preprocess data

data_loader = DataLoader()
data_loader.load_dataset()

# Split the data for evaluation
X_train, X_val, X_test, y_train, y_val, y_test = (
    data_loader.get_data_split()
)

# Preserve original training data before oversampling
X_train_original = X_train.copy()

# Oversample training data only
X_train, y_train = data_loader.oversample(X_train, y_train)

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nValidation class distribution:")
print(y_val.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# %% Fit Random Forest model

rf = RandomForestClassifier(
    class_weight='balanced',
    random_state=42,
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=5,
    max_features=0.5,
    n_jobs=-1
)

rf.fit(X_train, y_train)


# %% Experiment with different thresholds using validation data

y_val_prob = rf.predict_proba(X_val)[:, 1]

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

for threshold in thresholds:

    y_val_pred = (y_val_prob >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))
    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(y_val, y_val_pred)
    )
    print(
        "Macro F1:",
        f1_score(y_val, y_val_pred, average="macro")
    )
    print(
        "Stroke Precision:",
        precision_score(y_val, y_val_pred, zero_division=0)
    )
    print(
        "Stroke Recall:",
        recall_score(y_val, y_val_pred, zero_division=0)
    )
    print(
        "Stroke F1:",
        f1_score(y_val, y_val_pred, zero_division=0)
    )
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))


# %% Final test evaluation

selected_threshold = 0.4

y_test_prob = rf.predict_proba(X_test)[:, 1]

y_pred = (y_test_prob >= selected_threshold).astype(int)

print("\n========== FINAL TEST EVALUATION ==========")

print("Selected threshold:", selected_threshold)

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

print(
    "Stroke Precision:",
    precision_score(y_test, y_pred, zero_division=0)
)
print(
    "Stroke Recall:",
    recall_score(y_test, y_pred, zero_division=0)
)
print(
    "Stroke F1:",
    f1_score(y_test, y_pred, zero_division=0)
)

print("ROC-AUC:", roc_auc_score(y_test, y_test_prob))
print("Average Precision:", average_precision_score(y_test, y_test_prob))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Stroke (0)', 'Stroke (1)'],
    yticklabels=['No Stroke (0)', 'Stroke (1)'],
    annot_kws={"size": 14}
)

plt.title(
    f'Random Forest Confusion Matrix (Threshold = {selected_threshold})',
    fontsize=14
)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('Actual Label', fontsize=12)

plt.tight_layout()
plt.show()


# %% Create SHAP Explainer

# Use original training data as background
background = X_train_original.sample(
    n=min(100, len(X_train_original)),
    random_state=42
)

explainer = shap.TreeExplainer(
    rf,
    data=background,
    model_output="probability",
    feature_perturbation="interventional"
)

base_value = float(explainer.expected_value[1])

print("\nSHAP base value:", base_value)


# %% Calculate SHAP values for global explanation

# Select the first 100 observations from the held-out test set
X_shap = X_test.copy()
y_shap = y_test.copy()

print("\nCalculating global SHAP values...")

shap_values = explainer.shap_values(X_shap)

# Extract class 1 (stroke) contributions
stroke_shap_values = shap_values[:, :, 1]

print("SHAP output shape:", shap_values.shape)
print("Stroke SHAP shape:", stroke_shap_values.shape)
print("Explained class: Stroke (1)")


# %% Global SHAP sample information

print("\n========== GLOBAL SHAP SAMPLE ==========")

print("Total observations:", len(X_shap))
print("Positive class (stroke):", int((y_shap == 1).sum()))
print("Negative class (no stroke):", int((y_shap == 0).sum()))

print("Number of features:", X_shap.shape[1])
print("Feature representation: One-hot encoded columns")

print("Background sample size:", len(background))
print("Background random_state: 42")


# %% SHAP Verification

print("\n========== SHAP VERIFICATION ==========")

print("\nModel classes:", rf.classes_)
print("Explained class: Stroke (1)")
print("Class index:", list(rf.classes_).index(1))

print("\nInput shape:", X_shap.shape)
print("Number of input features:", X_shap.shape[1])
print("SHAP shape:", stroke_shap_values.shape)

print(
    "Feature names match:",
    list(X_shap.columns) == list(X_train_original.columns)
)

# Actual model probabilities
predicted_prob = rf.predict_proba(X_shap)[:, 1]

# Reconstruct probabilities using SHAP
reconstructed_prob = (
    base_value + stroke_shap_values.sum(axis=1)
)

# Calculate residuals
residual = predicted_prob - reconstructed_prob

print("\n----- First Test Sample -----")

print("Model-predicted probability:", predicted_prob[0])
print("Reconstructed probability:", reconstructed_prob[0])
print("Residual:", residual[0])

# Overall verification
max_residual = np.max(np.abs(residual))
mean_residual = np.mean(np.abs(residual))

print("\n----- Overall Verification -----")

print("Maximum absolute residual:", max_residual)
print("Mean absolute residual:", mean_residual)

verification_passed = np.allclose(
    predicted_prob,
    reconstructed_prob,
    atol=1e-5,
    rtol=1e-5
)

print("Verification passed:", verification_passed)

# Identify the observation with the largest residual
worst_index = np.argmax(np.abs(residual))

print("\n----- Largest Residual Observation -----")

print("Sample position (zero-based):", worst_index)
print("Anonymous row reference:", worst_index + 1)
print("Actual model probability:", predicted_prob[worst_index])
print("Reconstructed probability:", reconstructed_prob[worst_index])
print("Residual:", residual[worst_index])

print("\nTop feature contributions for this observation:")

worst_contributions = pd.DataFrame({
    "Feature": X_shap.columns,
    "SHAP value": stroke_shap_values[worst_index],
    "Feature value": X_shap.iloc[worst_index].values
})

worst_contributions["Absolute SHAP"] = (
    worst_contributions["SHAP value"].abs()
)

print(
    worst_contributions.sort_values(
        "Absolute SHAP",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\n=======================================")


# %% Global SHAP feature importance

mean_abs_shap = np.abs(stroke_shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_shap.columns,
    "Mean Absolute SHAP": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="Mean Absolute SHAP",
    ascending=False
).reset_index(drop=True)

importance_df.insert(
    0,
    "Rank",
    range(1, len(importance_df) + 1)
)

print("\n========== GLOBAL SHAP FEATURE IMPORTANCE ==========")

print("\nTop 5 features:")

print(
    importance_df.head(5).to_string(index=False)
)

# %% Global SHAP beeswarm summary plot

shap.summary_plot(
    stroke_shap_values,
    X_shap,
    feature_names=X_shap.columns,
    max_display=21,
    plot_size=(11, 9),
    show=False
)

plt.title(
    "Global SHAP Summary Plot - Stroke Prediction",
    fontsize=14
)

plt.show()
plt.close()


# %% Global SHAP mean absolute importance bar plot

shap.summary_plot(
    stroke_shap_values,
    X_shap,
    feature_names=X_shap.columns,
    plot_type="bar",
    max_display=21,
    plot_size=(10, 8),
    show=False
)

plt.title(
    "Global SHAP Feature Importance",
    fontsize=14
)

plt.show()
plt.close()


# %% Local explanation: First false negative

# Identify false negatives in the complete test set
false_negative_positions = np.where(
    (y_test.to_numpy() == 1) & (y_pred == 0)
)[0]

if len(false_negative_positions) == 0:
    raise ValueError("No false negative cases found in the test set.")

# Select the first false negative
case_pos = false_negative_positions[0]

X_patient_df = X_test.iloc[[case_pos]].copy()

actual_class = int(y_test.iloc[case_pos])
predicted_class = int(y_pred[case_pos])
patient_probability = float(y_test_prob[case_pos])

# Calculate SHAP values for the selected patient
patient_shap_output = explainer.shap_values(X_patient_df)

patient_shap = patient_shap_output[0, :, 1]

# Reconstruct local prediction
reconstructed_probability = (
    base_value + patient_shap.sum()
)

local_residual = (
    patient_probability - reconstructed_probability
)

print("\n========== SELECTED FALSE NEGATIVE ==========")

print("Anonymous test observation:", case_pos + 1)
print("Actual class:", actual_class)
print("Predicted class:", predicted_class)
print("Classification threshold:", selected_threshold)
print("Predicted stroke probability:", patient_probability)
print("Base value:", base_value)
print("Reconstructed probability:", reconstructed_probability)
print("Local residual:", local_residual)

print("Case type: False Negative (FN)")


# %% Local feature contribution table

contribution_df = pd.DataFrame({
    "Feature": X_patient_df.columns,
    "Patient value": X_patient_df.iloc[0].values,
    "SHAP value": patient_shap
})

contribution_df["Absolute SHAP"] = (
    contribution_df["SHAP value"].abs()
)

contribution_df = contribution_df.sort_values(
    "Absolute SHAP",
    ascending=False
)

top_3 = contribution_df.head(3).copy()

def get_direction(value):
    if value > 0:
        return "Increases stroke probability"
    elif value < 0:
        return "Decreases stroke probability"
    return "No contribution"

top_3["Direction"] = top_3["SHAP value"].apply(get_direction)

remaining_sum = contribution_df.iloc[3:]["SHAP value"].sum()

remaining_row = pd.DataFrame([{
    "Feature": "Remaining features combined",
    "Patient value": "-",
    "SHAP value": remaining_sum,
    "Absolute SHAP": abs(remaining_sum),
    "Direction": get_direction(remaining_sum)
}])

feature_table = pd.concat(
    [
        top_3[["Feature", "Patient value", "SHAP value", "Direction"]],
        remaining_row[["Feature", "Patient value", "SHAP value", "Direction"]]
    ],
    ignore_index=True
)

print("\n========== TOP LOCAL FEATURE CONTRIBUTIONS ==========")

print(feature_table.to_string(index=False))


# %% Generate local SHAP plots

# Create shared SHAP Explanation object
patient_explanation = shap.Explanation(
    values=patient_shap,
    base_values=base_value,
    data=X_patient_df.iloc[0].values,
    feature_names=X_patient_df.columns.tolist()
)

# Figure 2a: Static SHAP force plot
shap.force_plot(
    base_value,
    patient_shap,
    X_patient_df.iloc[0],
    matplotlib=True,
    show=False
)

plt.gcf().set_size_inches(16, 4)

plt.title(
    "Local SHAP Force Plot - False Negative",
    fontsize=13
)

plt.show()
plt.close()

# Optional: Save interactive force plot as HTML
shap.initjs()

interactive_force_plot = shap.force_plot(
    base_value,
    patient_shap,
    X_patient_df.iloc[0]
)

# Figure 2b: SHAP waterfall plot
shap.plots.waterfall(
    patient_explanation,
    max_display=10,
    show=False
)

plt.title(
    "Local SHAP Waterfall Plot - False Negative",
    fontsize=13
)

plt.show()
plt.close()

# %% Analyse global SHAP direction for the top five features

print("\n========== GLOBAL SHAP DIRECTION ANALYSIS ==========")

top_5_features = importance_df.head(5)["Feature"].tolist()

for feature in top_5_features:

    feature_values = X_shap[feature]
    feature_shap = stroke_shap_values[
        :, X_shap.columns.get_loc(feature)
    ]

    print(f"\nFeature: {feature}")

    # Binary / one-hot encoded feature
    if feature_values.nunique() <= 2:

        for value in sorted(feature_values.unique()):

            mask = feature_values == value

            print(
                f"Feature value {value}: "
                f"Count = {mask.sum()}, "
                f"Mean SHAP = {feature_shap[mask].mean():.6f}"
            )

    # Continuous numerical feature
    else:

        median_value = feature_values.median()

        low_mask = feature_values <= median_value
        high_mask = feature_values > median_value

        print("Median:", median_value)

        print(
            f"Lower/equal group: "
            f"Count = {low_mask.sum()}, "
            f"Mean SHAP = {feature_shap[low_mask].mean():.6f}"
        )

        print(
            f"Higher group: "
            f"Count = {high_mask.sum()}, "
            f"Mean SHAP = {feature_shap[high_mask].mean():.6f}"
        )

# %% Software versions

print("\n========== SOFTWARE VERSIONS ==========")

print("SHAP version:", shap.__version__)
print("Scikit-learn version:", sklearn.__version__)

print("\nAnalysis completed.")