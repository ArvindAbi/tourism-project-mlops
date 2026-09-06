# for data manipulation
import pandas as pd
# for checking file paths
import os

# Path of the dataset inside the GitHub repository
DATA_PATH = "tourism_project/data/tourism.csv"

# Expected columns as per the data dictionary
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome"
]

# Step 1: Check that the dataset file is present in the repo
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Please add tourism.csv to the data folder.")

# Step 2: Load the dataset
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully from the repository data folder.")

# Step 3: Validate that all expected columns are present
missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
if missing_cols:
    raise ValueError(f"Validation failed. Missing columns: {missing_cols}")
print("Validation passed. All expected columns are present.")

# Step 4: Print a short summary of the registered dataset
print("\n----- Dataset Summary -----")
print("Shape (rows, columns):", df.shape)
print("\nColumn data types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())
print("\nTarget variable (ProdTaken) distribution:")
print(df["ProdTaken"].value_counts())
print("\nFirst five rows:")
print(df.head())
print("\nDataset registered and validated successfully.")
