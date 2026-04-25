

# Ensemble DNN with 7 seeds - Parallel & Corrected Filename
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib  
from numpy import loadtxt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- 1. FIXED FILE LOADING ---
# Updated to match your 'ls' output: jan_025_train.csv
base_dir = os.path.dirname(os.path.abspath(__file__))
file_name = 'jan_025_train.csv' 
file_path = os.path.join(base_dir, file_name)

print(f"Checking for file at: {file_path}")

try:
    dataset = loadtxt(file_path, delimiter=',', skiprows=1)
    print(" File loaded successfully!")
except FileNotFoundError:
    print(f"Error: Still cannot find '{file_name}' in {base_dir}")
    raise

# --- 2. DATA PREPROCESSING ---
X = dataset[:, :36]
Y = dataset[:, 36].reshape(-1, 1)
X, Y = shuffle(X, Y)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

input_scaler = MinMaxScaler()
output_scaler = StandardScaler()
x_train = input_scaler.fit_transform(x_train)
x_test = input_scaler.transform(x_test)
y_train = output_scaler.fit_transform(y_train)
y_test_scaled = output_scaler.transform(y_test)

batch_size = 64

# --- 3. MODEL ARCHITECTURE ---
def build_model():
    inputs = Input(shape=(x_train.shape[1],))
    x = Dense(256, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(128, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(64, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(32, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
    return model

# --- 4. PARALLEL TRAINING ---
def train_single_seed(seed):
    tf.keras.utils.set_random_seed(seed)
    
    # Local data for workers
    t_data = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    v_data = tf.data.Dataset.from_tensor_slices((x_test, y_test_scaled)).batch(batch_size)
    
    model = build_model()
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, restore_best_weights=True)
    mc = ModelCheckpoint(f'Ensemble_DNN_seed{seed}.keras', monitor='val_loss', mode='min', save_best_only=True)
    
    model.fit(t_data, validation_data=v_data, epochs=1000, callbacks=[es, mc], verbose=0)
    p = model.predict(x_test, verbose=0)
    return p

seeds = [1, 42, 101, 202, 303, 404, 505]
print(f" Training ensemble in parallel on M4...")

results = joblib.Parallel(n_jobs=len(seeds))(
    joblib.delayed(train_single_seed)(s) for s in seeds
)

# --- 5. EVALUATION ---
all_preds = np.array(results)
ensemble_preds = np.mean(all_preds, axis=0)
ensemble_std = np.std(all_preds, axis=0)

ensemble_preds_rescaled = output_scaler.inverse_transform(ensemble_preds)
ensemble_std_rescaled = ensemble_std * output_scaler.scale_
y_test_rescaled = output_scaler.inverse_transform(y_test_scaled)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_rescaled, ensemble_preds_rescaled))
mae = mean_absolute_error(y_test_rescaled, ensemble_preds_rescaled)
r2 = r2_score(y_test_rescaled, ensemble_preds_rescaled)

print(f"\n Final Metrics:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Plot
plt.figure()
plt.errorbar(y_test_rescaled.flatten(), ensemble_preds_rescaled.flatten(), 
             yerr=ensemble_std_rescaled.flatten(), fmt='o', alpha=0.4)
plt.plot([y_test_rescaled.min(), y_test_rescaled.max()], 
         [y_test_rescaled.min(), y_test_rescaled.max()], 'r--')
plt.title('Final M4 Ensemble: Actual vs Predicted')
plt.show()