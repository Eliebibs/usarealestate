import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

print("Loading 20% sample datasets...")
train_df = pd.read_csv('USARealEstate/processed_train_20percent.csv')
test_df = pd.read_csv('USARealEstate/processed_test_20percent.csv')

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")

# Ensure both datasets have the same columns
common_columns = list(set(train_df.columns) & set(test_df.columns))
feature_columns = [col for col in common_columns if col not in ['price', 'price_log']]

X_train = train_df[feature_columns]
y_train = train_df['price']
X_test = test_df[feature_columns]
y_test = test_df['price']

# Create and train model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("\nTraining model...")
model.fit(X_train, y_train)

# Evaluate
print("\nEvaluating model...")
y_pred = model.predict(X_test)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate adjusted R-squared
n = len(y_test)
p = len(X_test.columns)
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("\n=== Model Performance Metrics ===")
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adjusted_r2:.4f}")

# Feature importance plot
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance['importance'])
plt.xticks(range(len(feature_importance)), feature_importance['feature'], rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('USARealEstate/feature_importance.png')
plt.close()

print("\nFeature importance plot saved as 'feature_importance.png'")
