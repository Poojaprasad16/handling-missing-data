# Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv('project2\pima-indians-diabetes.csv')

# Identify missing data (assumes that missing data is represented as NaN)
missing_counts = dataset.isnull().sum()

# Print the number of missing entries in each column
print("Number of missing entries in each column:")
print(missing_counts)
print()

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the DataFrame
imputer.fit(dataset)

# Apply the transform to the DataFrame
dataset = pd.DataFrame(imputer.transform(dataset), columns=dataset.columns)

# Print your updated matrix of features
print("Updated matrix of features (after handling missing data):")
print(dataset.head())
