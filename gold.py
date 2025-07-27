import pandas as pd
import numpy as np

data = pd.read_csv('/content/gold.csv')

print("First few rows of the dataset:")
print(data.head())

data.replace('na',np.nan, inplace=True)
data.dropna(inplace=True)
print(data.info())


data.drop_duplicates(inplace=True)
data['USD to INR'] = data['USD to INR'].abs()
data['Aver GLD per gram'] = data['Aver GLD per gram'].abs()
data['GLD INR per 10g'] = (data['GLD INR per 10g'].abs())
print(data)

import seaborn as sns
import matplotlib.pyplot as plt

data.hist(bins=50, figsize=(10, 8))
plt.show()

sns.pairplot(data)
plt.show()
plt.figure(figsize=(10, 8))
sns.heatmap(data.drop(columns=['Year']).corr(), annot=True,cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
columns_to_scale = ['USD to INR','Aver GLD per gram','GLD INR per 10g']

data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
print("Data after scaling:")
print(data.head())
print(data.tail())

from sklearn.feature_selection import SelectKBest,mutual_info_regression

target = 'Aver GLD per gram'
features = data.drop(columns=[target, 'Year'])


X = features
y = data[target]
mutual_info_selector = SelectKBest(mutual_info_regression, k=2)
X_kbest = mutual_info_selector.fit_transform(X, y)


selected_features_mutual_info = X.columns[mutual_info_selector.get_support()]
print("Selected features using Mutual Information:", selected_features_mutual_info)

X_selected = pd.DataFrame(X_kbest, columns=selected_features_mutual_info)
print("Data with selected features:")
print(X_selected.head())

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(X_selected, y,
test_size=0.2, random_state=42)


model_performance = {}
model_predictions = {}

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_pred)
model_performance['Linear Regression'] = lr_mse
model_predictions['Linear Regression'] = lr_pred

# Decision Tree Regressor
dt_model = DecisionTreeRegressor()
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)
model_performance['Decision Tree Regressor'] = dt_mse
model_predictions['Decision Tree Regressor'] = dt_pred

# RandomForest Regressor
rf_model = RandomForestRegressor()
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
model_performance['RandomForest Regressor'] = rf_mse
model_predictions['RandomForest Regressor'] = rf_pred

# Gradient Boosting Regressor
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_pred)
model_performance['Gradient Boosting Regressor'] = gb_mse
model_predictions['Gradient Boosting Regressor'] = gb_pred

# Support Vector Regressor (SVR)
svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_pred = svr_model.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_pred)
model_performance['Support Vector Regressor'] = svr_mse
model_predictions['Support Vector Regressor'] = svr_pred

# Display model performance
for model, mse in model_performance.items():
    print(f"{model}: Mean Squared Error = {mse}")

# View the dataset with predictions from the best model (choose model with the lowest MSE)
best_model = min(model_performance, key=model_performance.get)
print(f"\nBest Model: {best_model} with MSE = {model_performance[best_model]}")

# Adding predictions to the test dataset
X_test_with_predictions = X_test.copy()
X_test_with_predictions['Actual'] = y_test
X_test_with_predictions['Predicted'] = model_predictions[best_model]


# Display the dataset with actual and predicted values
print("\nDataset with Actual and Predicted values:")
print(X_test_with_predictions.head())


# Import necessary library for plotting
import matplotlib.pyplot as plt
import seaborn as sns


# Bar Plot: Model Performance Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_performance.keys()), y=list(model_performance.values()), palette='viridis')
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Model Performance Comparison')
plt.xticks(rotation=45)
plt.show()


models = ['Linear Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVR']
mse_values = [0.0000,0.0033,0.0033,0.0038,0.0085]

plt.figure(figsize=(8, 6))
sns.barplot(x=models, y=mse_values, palette="muted")
plt.title("Model Performance (MSE Comparison)")
plt.xlabel("Models")
plt.ylabel("Mean Squared Error (MSE)")
plt.xticks(rotation=45)
plt.show()


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(models, mse_values, marker='o', linestyle='-', color='blue')
plt.title("Model Performance (MSE Comparison)", fontsize=14)
plt.xlabel("Models", fontsize=12)
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.show()


# Horizontal bar plot
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(y=models, x=mse_values, palette="coolwarm")
plt.title("Model Performance (Horizontal MSE Comparison)", fontsize=14)
plt.xlabel("Mean Squared Error (MSE)", fontsize=12)
plt.ylabel("Models", fontsize=12)
plt.show()


# Pie chart for MSE comparison
plt.figure(figsize=(8, 6))
plt.pie(mse_values, labels=models, autopct='%1.1f%%', colors=sns.color_palette("pastel"), startangle=140)
plt.title("Model Performance (MSE Proportions)", fontsize=14)
plt.show()


# Box-and-whisker plot (assuming multiple MSE results per model)
import numpy as np

# Example cross-validated MSE values
mse_cross_val = {
    "Linear Regression": [0.053, 0.050, 0.054],
    "Decision Tree": [0.043, 0.045, 0.046],
    "Random Forest": [0.031, 0.032, 0.033],
    "Gradient Boosting": [0.027, 0.029, 0.028],
    "SVR": [0.038, 0.039, 0.041]
}

plt.figure(figsize=(8, 6))
sns.boxplot(data=list(mse_cross_val.values()), palette="Set3")
plt.xticks(ticks=np.arange(len(models)), labels=models, rotation=45)
plt.title("Model Performance (MSE Distribution)", fontsize=14)
plt.xlabel("Models", fontsize=12)
plt.ylabel("Mean Squared Error (MSE)", fontsize=12)
plt.show()
