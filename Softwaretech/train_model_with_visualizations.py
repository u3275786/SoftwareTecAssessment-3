import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Step 1: Reading the Data
file_path = 'NFLX.csv.xls'  # Update with your file path
data = pd.read_csv(file_path)
data_cleaned = data.drop_duplicates()

# Step 2: Define target variable and predictors
target_variable = 'Close'
predictor_variables = ['Open', 'High', 'Low', 'Adj Close', 'Volume']

# Step 3: Visualize the distribution of the Close Prices
plt.hist(data_cleaned['Close'], bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Close Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.show()

# Step 4: Data Exploration
print(data_cleaned.info())
print(data_cleaned.describe())

# Step 5: Visual Exploratory Data Analysis (EDA)
data_cleaned.hist(bins=30, figsize=(10, 8), edgecolor='black')
plt.suptitle('Histograms of Continuous Variables')
plt.show()

# Step 6: Outlier Analysis using Boxplots
columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
plt.figure(figsize=(15, 10))
for i, col in enumerate(columns, 1):
    plt.subplot(2, 3, i)
    plt.boxplot(data_cleaned[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Step 7: Correlation Analysis using Heatmap
numeric_data = data_cleaned.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Matrix with Annotations')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Step 8: Model Preparation
selected_features = ['Open', 'High', 'Low', 'Adj Close', 'Volume']
X = data_cleaned[selected_features]
y = data_cleaned['Close']

# Step 9: Split data and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 10: Train and Evaluate Multiple Models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42),
    'AdaBoost': AdaBoostRegressor(random_state=42),
    'KNN': KNeighborsRegressor()
}

mse_results = {}
mae_results = {}
r2_results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = model.score(X_test_scaled, y_test)
    
    mse_results[name] = mse
    mae_results[name] = mae
    r2_results[name] = r2

# Step 11: Visualizing Model Performance

# Create DataFrame for performance metrics
performance_df = pd.DataFrame({
    'MSE': mse_results,
    'MAE': mae_results,
    'R2': r2_results
})

# Visualize MSE for each model
plt.figure(figsize=(10, 6))
sns.barplot(x=performance_df.index, y='MSE', data=performance_df, palette='Blues_d')
plt.title('Mean Squared Error (MSE) of Different Models')
plt.ylabel('MSE')
plt.xlabel('Models')
plt.show()

# Visualize MAE for each model
plt.figure(figsize=(10, 6))
sns.barplot(x=performance_df.index, y='MAE', data=performance_df, palette='Greens_d')
plt.title('Mean Absolute Error (MAE) of Different Models')
plt.ylabel('MAE')
plt.xlabel('Models')
plt.show()

# Visualize R-squared for each model
plt.figure(figsize=(10, 6))
sns.barplot(x=performance_df.index, y='R2', data=performance_df, palette='Reds_d')
plt.title('R-squared (R2) of Different Models')
plt.ylabel('R2 Score')
plt.xlabel('Models')
plt.show()

# Step 12: Print Model Performance Summary
print("Model Performance Summary:")
print(performance_df)

# Selecting the best model based on MSE
best_model_name = performance_df['MSE'].idxmin()
print(f"The best model based on MSE is: {best_model_name}")
