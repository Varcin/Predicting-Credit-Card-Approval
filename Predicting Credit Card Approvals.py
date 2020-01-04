# 2019-08-17
# Predicting Credit Card Approval
# Data source: http://archive.ics.uci.edu/ml/datasets/credit+approval

# Import Packages
import pandas as pd

# Load dataset
cc_apps = pd.read_csv("datasets/cc_approvals.data", header = None)
cc_apps = pd.read_csv("C:/Users/Varcin/Dropbox/DS Projects/Python/Predicting Credit Card Approval/crx.data", header = None)

# Inspect data
cc_apps.head()

# Print summary statistics
cc_apps_description = cc_apps.describe()
print(cc_apps_description)

print("\n")

# Print DataFrame information
cc_apps_info = cc_apps.info()
print(cc_apps_info)

print("\n")

# Inspect missing values in the dataset
cc_apps.tail()

# The missing values in the dataset are labeled with '?', which can be seen in the last cell's output.
# temporarily replace these missing value question marks with NaN

# Import numpy
import numpy as np

# Inspect missing values in the dataset
cc_apps.tail(17)

# Replace the '?'s with NaN
cc_apps = cc_apps.replace("?", np.nan)

# Inspect the missing values again
cc_apps.tail(17)



# Impute the missing values with mean imputation
cc_apps.fillna(cc_apps.mean(), inplace=True)

# Count the number of NaNs in the dataset to verify
cc_apps.isnull()


# Iterate over each column of cc_apps
for col in cc_apps:
    # Check if the column is of object type
    if cc_apps[col].dtypes == 'object':
        # Impute with the most frequent value
        cc_apps = cc_apps.fillna(cc_apps[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
cc_apps.isnull().sum()



# Things to do before runnning the model
# Convert the non-numeric data into numeric.
# Split the data into train and test sets.
# Scale the feature values to a uniform range.


#  many machine learning models (like XGBoost) (and especially the ones developed using scikit-learn)
# require the data to be in a strictly numeric format. We will do this by using a technique called label encoding.
