import numpy as np
from numpy import loadtxt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load dataset
dataset = loadtxt('C:\\feb_025_train_all_param.csv', delimiter=',', skiprows=1)
X = dataset[:, 0:36]
Y = dataset[:, 36].reshape(-1, 1)

# Shuffle and split
X, Y = shuffle(X, Y, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Use robust scaling to reduce impact of outliers
input_scaler = RobustScaler()
output_scaler = RobustScaler()

x_train_scaled = input_scaler.fit_transform(x_train)
x_test_scaled = input_scaler.transform(x_test)

y_train_scaled = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)

# GridSearch to find best parameters
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance']
}

knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(x_train_scaled, y_train_scaled.ravel())

best_knn = grid_search.best_estimator_
print(f"Best KNN parameters: {grid_search.best_params_}")

# Evaluate model
train_pred_scaled = best_knn.predict(x_train_scaled)
test_pred_scaled = best_knn.predict(x_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train_scaled, train_pred_scaled))
test_rmse = np.sqrt(mean_squared_error(y_test_scaled, test_pred_scaled))
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")

# Inverse transform predictions
train_pred = output_scaler.inverse_transform(train_pred_scaled.reshape(-1, 1))
test_pred = output_scaler.inverse_transform(test_pred_scaled.reshape(-1, 1))
y_test_orig = output_scaler.inverse_transform(y_test_scaled)

# Metrics on original scale
mae = mean_absolute_error(y_test_orig, test_pred)
r2 = r2_score(y_test_orig, test_pred)
rmse = np.sqrt(mean_squared_error(y_test_orig, test_pred))

print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# Residuals
residuals = y_test_orig - test_pred

# Plot predictions vs actual
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, test_pred, c='blue', alpha=0.6, label='Predicted vs Actual')
plt.plot([y_test_orig.min(), y_test_orig.max()], [y_test_orig.min(), y_test_orig.max()], 'r--', label='Ideal fit')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('KNN Predictions vs Actual')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_prediction_feb_025_improved.pdf")
plt.show()

# Residual plot
plt.figure(figsize=(10, 4))
plt.scatter(test_pred, residuals, alpha=0.6)
plt.axhline(0, color='r', linestyle='--')
plt.title("Residuals vs Predicted")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_feb_025_residuals_plot.pdf")
plt.show()

# Save predictions
np.savetxt("knn_improved_feb_025_x_test_predictions.csv", test_pred, delimiter=",")
np.savetxt("knn_improved_feb_025_y_test.csv", y_test_orig, delimiter=",")

# Save model and scalers
joblib.dump(best_knn, "knn_model_improved.pkl")
joblib.dump(input_scaler, "input_scaler_improved.pkl")
joblib.dump(output_scaler, "output_scaler_improved.pkl")

