import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Load the dataset
file_path = 'NFLX.csv.xls'  # Update this path with your file location
data = pd.read_csv(file_path)
data_cleaned = data.drop_duplicates()

# Define all 6 features for visualization purposes
# Swap the positions of 'Volume' and 'Close'
all_features = ['Open', 'High', 'Low', 'Adj Close', 'Close', 'Volume']

# Step 2: Visualizations (ALL 6 FEATURES)

# Histogram of Close Prices
plt.hist(data_cleaned['Close'], bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Close Prices')
plt.xlabel('Close Price')
plt.ylabel('Frequency')
plt.show()

# Histograms for each feature (Open, High, Low, Adj Close, Close, Volume)
data_cleaned[['Open', 'High', 'Low', 'Adj Close', 'Close', 'Volume']].hist(bins=30, figsize=(10, 8), edgecolor='black')
plt.suptitle('Histograms of Features (Open, High, Low, Adj Close, Close, Volume)')
plt.show()

# Scatter plots between each feature and the target variable (Close)
plt.figure(figsize=(20, 12))
for i, col in enumerate(all_features[:-1], 1):  # Visualizing all features except 'Volume'
    plt.subplot(2, 3, i)  # 2 rows, 3 columns layout for 5 features
    plt.scatter(data_cleaned[col], data_cleaned['Close'], alpha=0.5)
    plt.title(f'{col} vs Close')
    plt.xlabel(col)
    plt.ylabel('Close')
plt.tight_layout()
plt.show()

# Correlation Heatmap for all features
correlation_matrix = data_cleaned[all_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Matrix of All Features')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Boxplots for outlier detection for all features (Open, High, Low, Adj Close, Close, Volume)
plt.figure(figsize=(15, 10))
for i, col in enumerate(all_features, 1):
    plt.subplot(2, 3, i)  # 2 rows, 3 columns for 6 features (Open, High, Low, Adj Close, Close, Volume)
    plt.boxplot(data_cleaned[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

# Step 3: Define predictor variables (only 4) and target variable (Close)
# Use only 4 features for model training: 'Open', 'High', 'Low', 'Volume'
predictor_variables = ['Open', 'High', 'Low', 'Volume']  # Exclude 'Adj Close'
X = data_cleaned[predictor_variables]
y = data_cleaned['Close']

# Step 4: Scale the predictor variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train the Linear Regression model (ONLY USE 4 FEATURES)
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_scaled, y)

# Step 6: Print the model's coefficients and intercept for debugging
print("Model Coefficients:", linear_regression_model.coef_)
print("Model Intercept:", linear_regression_model.intercept_)

# Step 7: Save the trained model and scaler using joblib
joblib.dump(linear_regression_model, 'linear_regression_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully!")
