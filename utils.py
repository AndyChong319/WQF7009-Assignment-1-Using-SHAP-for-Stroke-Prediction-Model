import pandas as pd 
# Makes sure we see all columns
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

class DataLoader():

    def __init__(self):
        self.data = None
        self.feature_columns = None
        self.bmi_median = None

    def load_dataset(self, path="data/healthcare-dataset-stroke-data.csv"):
        self.data = pd.read_csv(path)

    def preprocess_data(self):
        """
        Kept for compatibility with the original project.
        Preprocessing is now performed inside get_data_split()
        AFTER splitting to prevent data leakage.
        """
        pass

    def get_data_split(self):

        # Separate predictors and target
        X = self.data.drop(columns=["id", "stroke"])
        y = self.data["stroke"]

        # First split: 80% training, 20% temporary data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        # Second split: 10% validation, 10% testing
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.50,
            random_state=42,
            stratify=y_temp
        )

        # Calculate BMI median using TRAINING DATA ONLY
        self.bmi_median = X_train["bmi"].median()

        # Apply the same median to all three datasets
        X_train = X_train.copy()
        X_val = X_val.copy()
        X_test = X_test.copy()

        X_train["bmi"] = X_train["bmi"].fillna(self.bmi_median)
        X_val["bmi"] = X_val["bmi"].fillna(self.bmi_median)
        X_test["bmi"] = X_test["bmi"].fillna(self.bmi_median)

        # One-hot encoding
        categorical_cols = [
            "gender",
            "ever_married",
            "work_type",
            "Residence_type",
            "smoking_status"
        ]

        X_train = pd.get_dummies(
            X_train,
            columns=categorical_cols,
            dtype=int
        )

        # Save training feature names and order
        self.feature_columns = X_train.columns.tolist()

        # Encode validation and test using the same feature columns
        X_val = pd.get_dummies(
            X_val,
            columns=categorical_cols,
            dtype=int
        )

        X_test = pd.get_dummies(
            X_test,
            columns=categorical_cols,
            dtype=int
        )

        X_val = X_val.reindex(
            columns=self.feature_columns,
            fill_value=0
        )

        X_test = X_test.reindex(
            columns=self.feature_columns,
            fill_value=0
        )

        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test
        )

    def oversample(self, X_train, y_train):

        oversample = RandomOverSampler(
            sampling_strategy="minority",
            random_state=42
        )

        x_np, y_np = oversample.fit_resample(
            X_train,
            y_train
        )

        x_over = pd.DataFrame(
            x_np,
            columns=X_train.columns
        )

        y_over = pd.Series(
            y_np,
            name=y_train.name
        )

        return x_over, y_over