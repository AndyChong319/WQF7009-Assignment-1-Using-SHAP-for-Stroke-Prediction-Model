# %% Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from utils import DataLoader


# %% Load data
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data

# %% Show head
print("Dataset dimensions:", data.shape)
print("Number of rows:", data.shape[0])
print("Number of columns:", data.shape[1])
data.head()

# %% Show general statistics
data.info()
dup = data.duplicated().sum()
print("\nDuplicate data:", dup, "\n")
print("Missing data:\n\n", data.isna().sum())

# %% Stroke class count

stroke_counts = data['stroke'].value_counts().sort_index()

print("Stroke Class Count:")
print(stroke_counts)

print("\nStroke Class Percentage:")
print(data['stroke'].value_counts(normalize=True).sort_index() * 100)

# %% Descriptive statistics

numerical_cols = ['age', 'avg_glucose_level', 'bmi']

# Mean and Standard Deviation
mean_sd = data[numerical_cols].agg(['mean', 'std']).T

# Median and Interquartile Range
median_iqr = data[numerical_cols].agg(
    ['median',
     lambda x: x.quantile(0.25),
     lambda x: x.quantile(0.75)]
).T

median_iqr.columns = ['Median', 'Q1', 'Q3']

# Combine results
stats = pd.concat([mean_sd, median_iqr], axis=1)

print("Descriptive Statistics:")
print(stats.round(2),"\n")

# %% Combined categorical distributions

categorical_cols = [
    'gender',
    'hypertension',
    'heart_disease',
    'ever_married',
    'work_type',
    'Residence_type',
    'smoking_status',
    'stroke'
]

results = []

for col in categorical_cols:
    counts = data[col].value_counts(dropna=False)
    percentages = data[col].value_counts(
        normalize=True, dropna=False
    ) * 100

    for category in counts.index:
        results.append({
            'Variable': col,
            'Category': category,
            'Count (n)': counts[category],
            'Percentage (%)': percentages[category]
        })

categorical_summary = pd.DataFrame(results)

print(categorical_summary.round(2))

# %% Show histogram for all columns
columns = data.columns
for col in columns:
    print("col: ", col)
    plt.figure(figsize=(8, 5))
    data[col].hist(bins=20, edgecolor='black')

    plt.title(f'Histogram of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    plt.grid(False)
    plt.tight_layout()
    plt.show()

# missing value plot
missing = data.isna().sum()
missing = missing[missing > 0]

plt.figure(figsize=(7, 4))

missing.plot(kind='bar', color='orange', edgecolor='black')

plt.title('Missing Values by Variable')
plt.xlabel('Variable')
plt.ylabel('Number of Missing Values')
plt.tight_layout()
plt.show()

# Age Distribution by Stroke Status
plt.figure(figsize=(7, 5))

sns.boxplot(
    data=data,
    x='stroke',
    y='age'
)

plt.title('Age Distribution by Stroke Status')
plt.xlabel('Stroke (0 = No, 1 = Yes)')
plt.ylabel('Age')

plt.tight_layout()
plt.show()

# Stroke Status with agl, bmi
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(
    data=data,
    x='stroke',
    y='avg_glucose_level',
    ax=axes[0]
)
axes[0].set_title('Average Glucose Level by Stroke Status')
axes[0].set_xlabel('Stroke (0 = No, 1 = Yes)')

sns.boxplot(
    data=data,
    x='stroke',
    y='bmi',
    ax=axes[1]
)

axes[1].set_title('BMI by Stroke Status')
axes[1].set_xlabel('Stroke (0 = No, 1 = Yes)')

plt.tight_layout()
plt.show()

# %% Show preprocessed dataframe
data_loader.preprocess_data()
data_loader.data.head()