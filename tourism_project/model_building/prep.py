# for data manipulation
import pandas as pd
# for splitting the data into train and test sets
from sklearn.model_selection import train_test_split

# Load the dataset directly from the repository data folder
DATA_PATH = "tourism_project/data/tourism.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# Data cleaning
# Drop the unnecessary columns which do not help in prediction
# 'Unnamed: 0' is just a row index and 'CustomerID' is a unique identifier
df.drop(columns=["Unnamed: 0", "CustomerID"], inplace=True)

# Fix the typo in the Gender column ('Fe Male' should be 'Female')
df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# 'Unmarried' and 'Single' mean the same, so merge them into one category
df["MaritalStatus"] = df["MaritalStatus"].replace("Unmarried", "Single")

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)

print("Data cleaning completed. Shape after cleaning:", df.shape)

# Define the target variable
target_col = "ProdTaken"

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split (80% train, 20% test)
# stratify=y keeps the same class ratio in both splits since the target is imbalanced
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save the splits locally as CSV files
# The GitHub Actions workflow passes these files to the next job as an artifact
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train and test splits saved locally.")
print("Xtrain:", Xtrain.shape, "| Xtest:", Xtest.shape)
print("ytrain:", ytrain.shape, "| ytest:", ytest.shape)
