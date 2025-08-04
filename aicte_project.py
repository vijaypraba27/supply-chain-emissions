
"""## Importing Libraries"""

import pandas as pd
import numpy as np


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error ,mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import joblib

from warnings import simplefilter
simplefilter(action='ignore')
"""## LOAD DATASET"""

excel_file = pd.ExcelFile(r"C:\Users\T M\Desktop\GHG_Emission_dataset.xlsx",engine='openpyxl')
print(excel_file.sheet_names)
years=range(2010,2017)

years[0]

years[1]

df_1=pd.read_excel(excel_file,sheet_name=f'{years[0]}_Detail_Commodity')
df_1.head()

df_2=pd.read_excel(excel_file,sheet_name=f'{years[0]}_Detail_Industry')
df_2.head()

all_data = []

for year in years:
    try:
        # Read the 'Detail_Commodity' sheet
        # Make sure the sheet name is exactly as it appears in your Excel file
        df_com = pd.read_excel(r"C:\Users\T M\Desktop\GHG_Emission_dataset.xlsx", sheet_name=f'{year}_Detail_Commodity')

        # Read the 'Detail_Industry' sheet
        df_ind = pd.read_excel(r"C:\Users\T M\Desktop\GHG_Emission_dataset.xlsx", sheet_name=f'{year}_Detail_Industry')

        # Add 'Source' column to differentiate data origins
        df_com['Source'] = 'Commodity'
        df_ind['Source'] = 'Industry'

        # Add 'Year' column to both DataFrames
        df_com['Year'] = year
        df_ind['Year'] = year

        # Strip whitespace from column names for consistency
        df_com.columns = df_com.columns.str.strip()
        df_ind.columns = df_ind.columns.str.strip()

        # Rename columns for consistency across commodity and industry data
        df_com.rename(columns={'Commodity Code': 'Code', 'Commodity Name': 'Name'}, inplace=True)
        df_ind.rename(columns={'Industry Code': 'Code', 'Industry Name': 'Name'}, inplace=True)

        # Concatenate commodity and industry data for the current year
        # and append to the all_data list
        all_data.append(pd.concat([df_com, df_ind], ignore_index=True))


    except Exception as e:
        print(f"An unexpected error occurred while processing year {year}: {e}")

all_data[3]

len(all_data)

df=pd.concat(all_data,ignore_index=True)
df.head(11)

#shape
df.shape

#info()
df.info()

df.columns

#finding missing value
df.isnull().sum()/df.shape[0]*100

#finding duplicates
df.duplicated().sum()

#identifiying garbage values
for i in df.select_dtypes(include="object").columns:
  print(df[i].value_counts())
  print("")

"""## Data Preprocessing"""

#descriptive statistics
df.describe().T

df.describe(include="object").T

#histrogram to understand the distribution
for i in df.select_dtypes(include="number").columns:
  sns.histplot(data=df,x=i,color='aqua',bins=50)
  plt.show()

# Check caterical value
print(df['Substance'].value_counts())

print(df['Unit'].value_counts())

# Checking unique values in 'Unit'
print(df['Unit'].unique())

print(df['Source'].value_counts())

print(df['Source'].unique())

df['Substance'].unique()

df.Code.unique()

df.Name.unique()

len(df.Name.unique())

"""## Top 10 Emmiting Industry"""

top_emitters=df[['Name','Supply Chain Emission Factors with Margins']].groupby('Name').mean().sort_values('Supply Chain Emission Factors with Margins',ascending=False).head(10)

# Resetting index for betterplotting
top_emitters=top_emitters.reset_index()
top_emitters

plt.figure(figsize=(10, 6))

# Barplot for top emitting industries
sns.barplot(
    x='Supply Chain Emission Factors with Margins',
    y='Name',
    data=top_emitters,
    palette='viridis'  # Corrected palette name
)

# Add ranking labels (1, 2, 3...) next to bars
for i, (value, name) in enumerate(zip(top_emitters['Supply Chain Emission Factors with Margins'], top_emitters['Name']), start=1):
    plt.text(
        x=value + 0.01,  # slightly offset to the right of the bar
        y=i - 1,         # match the y position of the bar
        s=f'#{i}',
        va='center',
        fontsize=11,
        fontweight='bold',
        color='brown'
    )

# Title and axis labels
plt.title('Top 10 Emitting Industries', fontsize=14, fontweight='bold')
plt.xlabel('Emission Factor (kg CO2e/unit)')
plt.ylabel('Industry')

# Add grid lines
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Prevent overlap
plt.tight_layout()

# Show plot
plt.show()

# Count plot for Substance
plt.figure(figsize=(5,3))
sns.countplot(x=df['Substance'])
plt.title('Count Plot for Substance')
plt.xlabel('Substance')
plt.ylabel('Count')
plt.show()

# Count plot for Unit
plt.figure(figsize=(5,3))
sns.countplot(x=df['Unit'])
plt.title('Count Plot for Unit')
plt.xlabel('Unit')
plt.ylabel('Count')
plt.show()

# Count plot for Source
plt.figure(figsize=(5,3))
sns.countplot(x=df['Source'])
plt.title('Count Plot for Source(Industry vs Commodity)')
plt.xlabel('Source')
plt.ylabel('Count')
plt.show()

"""## Box Plot Analysis of [Variable Name]"""

#Boxplot -to-identify Outliers
for i in df.select_dtypes(include="number").columns:
    sns.boxplot(x=df[i].dropna(), color='red')
    plt.title(f'Boxplot of {i}')
    plt.xlabel(i)
    plt.show()

df.columns

#Scatter plot to understand the relationship
for i in ['Code', 'Name', 'Substance', 'Unit',
       'Supply Chain Emission Factors without Margins',
       'Margins of Supply Chain Emission Factors',
       'Supply Chain Emission Factors with Margins', 'Unnamed: 7',
       'DQ ReliabilityScore of Factors without Margins',
       'DQ TemporalCorrelation of Factors without Margins',
       'DQ GeographicalCorrelation of Factors without Margins',
       'DQ TechnologicalCorrelation of Factors without Margins',
       'DQ DataCollection of Factors without Margins', 'Source', 'Year']:
       sns.scatterplot(data=df,x=i,y='Supply Chain Emission Factors without Margins')
       plt.show()

"""## Multivariate Analysis"""

#correlation with heatmap to interpret the relation and multicolliniarity
plt.figure(figsize=(12,8))
sns.heatmap(df.select_dtypes(include=np.number).corr(),annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

"""## Missing value treatments"""

#choose the method inputing missing value
#like mean,median,mode or KNNIputer

for col in df.select_dtypes(include='object').columns:
    mode_value = df[col].mode()[0]  # Get the most frequent value
    df[col] = df[col].fillna(mode_value)

impute=KNNImputer()
# Create an imputer for categorical columns
imputer = SimpleImputer(strategy='most_frequent')

# Loop through object columns
for col in df.select_dtypes(include="object").columns:
    df[col] = imputer.fit_transform(df[[col]]).ravel()

"""## Outliers treatments"""

#decide whether to do outliers treaments or not,if do how?

def wisker(df, col, method='detect', verbose=True):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    if verbose:
        print(f"Column: {col}")
        print(f"Lower bound: {lower:.2f}, Upper bound: {upper:.2f}")
        print(f"Outliers count: {outliers.shape[0]}\n")

    if method == 'cap':
        df[col] = np.where(df[col] < lower, lower,
                  np.where(df[col] > upper, upper, df[col]))
        return df

wisker(df, 'Supply Chain Emission Factors without Margins')

df.columns

columns_for_whiskers = [
    'Supply Chain Emission Factors without Margins',
    'Margins of Supply Chain Emission Factors',
    'Supply Chain Emission Factors with Margins',
    'DQ ReliabilityScore of Factors without Margins',
    'DQ TemporalCorrelation of Factors without Margins',
    'DQ GeographicalCorrelation of Factors without Margins',
    'DQ TechnologicalCorrelation of Factors without Margins',
    'DQ DataCollection of Factors without Margins'
]

for col in columns_for_whiskers:
    if col in df.columns:
        sns.boxplot(y=df[col])
        plt.title(f'Boxplot (Whiskers) for: {col}')
        plt.show()

"""Let's prepare the data for a classification task to generate the `y_test` and `y_pred` variables needed for the confusion matrix. I will use the 'Source' column as the target variable for this example."""

# Drop non-numeric and irrelevant columns for this example
df_for_model = df.drop(['Code', 'Name', 'Substance', 'Unit'], axis=1)

# Handle the 'Unnamed: 7' column if it still exists and has NaNs (although we dropped it earlier, good to be safe)
if 'Unnamed: 7' in df_for_model.columns:
  df_for_model.drop('Unnamed: 7', axis=1, inplace=True)


# Separate features (X) and target (y)
X = df_for_model.drop('Source', axis=1)
y = df_for_model['Source']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Train a simple Logistic Regression model
model = LogisticRegression(max_iter=1000) # Increase max_iter if convergence warnings occur
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

"""Now that we have `y_test` and `y_pred`, we can plot the confusion matrix."""

# Plot confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

"""## Duplicates and Garbege values"""

#check for uplicate if we have any uniquie column in the data set
#clean the garbege value

df.drop_duplicates()

"""## Normalize Features"""

# Define features (X) and target (y)
# Drop non-numeric and irrelevant columns for this example
X = df.drop(columns=['Supply Chain Emission Factors with Margins', 'Code', 'Name', 'Substance', 'Unit', 'Unnamed: 7', 'Source'])
y = df['Supply Chain Emission Factors with Margins']

X.head()

y.head()

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

X_scaled[0].min(),X_scaled[0].max()

np.round(X_scaled.mean()),np.round(X_scaled.std())

"""## Divide the data into train and test"""

X.shape

X_train,X_test,y_train,y_test=train_test_split(X_scaled,y,test_size=0.2,random_state=42)

X_train.shape

X_test.shape

"""## Select the model for training"""

# Create the model
RF_model = RandomForestRegressor(random_state=42)

RF_model.fit(X_train,y_train)

"""## Prediction and evalution"""

# Making prediction on the test set
RF_y_pred=RF_model.predict(X_test)

RF_y_pred[:40]

# Calculating meand squared Error(MSE)
RF_mse=mean_squared_error(y_test,RF_y_pred)

# Calculating Root mean squared Error(RMSE)
RF_rmse=np.sqrt(RF_mse)
RF_r2=r2_score(y_test,RF_y_pred)

print(f"MSE: {RF_mse}")
print(f"RMSE: {RF_rmse}")
print(f"R2: {RF_r2}")

# Initialize Linear Regression model
LR_model = LinearRegression()

# Fit model on training data
LR_model.fit(X_train, y_train)

# Predict on test data
LR_y_pred = LR_model.predict(X_test)

# Calculate metrics
LR_mse = mean_squared_error(y_test, LR_y_pred)        # Mean Squared Error
LR_rmse = np.sqrt(LR_mse)
LR_r2 = r2_score(y_test, LR_y_pred)

# Print metrics
print(f"MSE: {LR_mse}")
print(f"RMSE: {LR_rmse}")
print(f"R2: {LR_r2}")

"""## Hyperparameter tuning"""

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

# Perform Grid Search with 3-fold Cross-Validation
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    n_jobs=-1
)

# Fit the grid search on training data
grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Use the best model to make predictions on the test set
y_pred_best = best_model.predict(X_test)

# Calculate evaluation metrics
HP_mse = mean_squared_error(y_test, y_pred_best)
HP_rmse = np.sqrt(HP_mse)
HP_r2 = r2_score(y_test, y_pred_best)

# Print metrics
print(f"MSE: {HP_mse}")
print(f"RMSE: {HP_rmse}")
print(f"R2: {HP_r2}")

"""## Model Accuraccy Comparison: RF vs LR"""

# Create a comparative DataFrame for all models
results = {
    'Model': ['Random Forest (Default)', 'Linear Regression', 'Random Forest (Tuned)'],
    'MSE': [RF_mse, LR_mse, HP_mse],
    'RMSE': [RF_rmse, LR_rmse, HP_rmse],
    'R2': [RF_r2, LR_r2, HP_r2]
}

# Convert to DataFrame and display
comparison_df = pd.DataFrame(results)
print(comparison_df)

# Create a directory to save the models if if doesn't exist
import os
os.makedirs("models",existss_ok=True)

# Save model and encoders
joblib.dump(best_model, 'models/LR_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
