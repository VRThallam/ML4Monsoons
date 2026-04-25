ML4Monsoons – Monsoon Rainfall Prediction
Problem Statement
The objective of this project is to predict monsoon rainfall (June–September) in advance using climate variables available in earlier months (January, February, March).
The goal is to estimate rainfall values as close as possible to IMD (Indian Meteorological Department) observations using multiple machine learning models and spatial resolutions.

Aim
Identify the best model
Identify the best spatial resolution (0.25° vs 1°)
Identify the best input configuration (all parameters vs single parameter)
Identify the best prediction month (Jan / Feb / Mar)

Final objective: Build a reliable early rainfall prediction system

Why Predict from Jan–Mar?
Monsoon rainfall is influenced by large-scale climate patterns that begin months earlier.
January–March contain early atmospheric signals
Helps in long-range forecasting
Useful for:
Agriculture planning 
Water resource management 
Disaster preparedness 

Approach Overview
Data Used
Gridded climate dataset
Two spatial resolutions:
0.25° (high resolution)
1° (coarse resolution)

Features
Precipitation (PR)
Geopotential Height (GZ)
Temperature (TR)

Target
IMD rainfall values

Models Used
We implemented and compared:

KNN
Ridge Regression
Random Forest (RF)
Dense Neural Network
Ensemble DNN
Optuna-based Neural Network
EML (Ensemble Machine Learning)
Stacked Ensemble (with Quantile Features)

Hyperparameters Used
KNN
n_neighbors: 3, 5, 7, 9, 11 (GridSearch)
weights: uniform / distance
Ridge Regression
alpha = 1.0

Random Forest (RF)
n_estimators: 50–300
max_depth: 10–None
min_samples_split: 2, 5, 10
min_samples_leaf: 1, 2, 4
max_features: auto, sqrt, log2

Dense Neural Network
Layers: 256 → 128 → 64 → 32
Activation: ReLU
Dropout: 0.2
L2 Regularization: 0.0005
Optimizer: Adam
Learning rate: 1e-4

Ensemble DNN
Same architecture as Dense model
Multiple seeds: [1, 42, 101, 202, 303, 404, 505]
Final prediction = average of models

Optuna Model
Units per layer: tuned (16–512)
Dropout: 0.1–0.4
L2 regularization: 1e-5 → 1e-3
Learning rate: 1e-5 → 1e-3
Trials: 30

EML 
Base models:
KNN
Ridge
Gradient Boosting
Meta model:
Linear Regression

Stacked Ensemble
Base models:
KNN, Ridge, XGBoost
Additional features:
Median prediction (Q50)
Uncertainty band (Q90 - Q10)
Meta model:
Linear Regression / XGBoost

Experiment Flow (What Was Done)
Step 1: Model Comparison (1° Resolution)
Trained all models on Jan, Feb, Mar
Best Model → Optuna
Step 2: Parameter Comparison (1°)
All parameters vs single parameter
Best → All parameters
Step 3: Month Comparison (1°)
Jan vs Feb vs Mar
Best → March
Step 4: Model Comparison (0.25° Resolution)
Trained all models
Best Model → Random Forest (RF)
Step 5: Parameter Comparison (0.25°)
All parameters vs single
Best → All parameters
Step 6: Month Comparison (0.25°)
Best → March
Step 7: Resolution Comparison
0.25° vs 1°
Best → 0.25° (higher spatial accuracy)

Important Observation
Using:
RF
0.25°
All parameters
Predictions showed consistent underestimation

Final Model Decision
We tested all models again on test data (2019–2023):
EML had the lowest bias

Why EML Performed Better than RF?
EML combines strengths of multiple models:

KNN → captures local patterns
Ridge → handles linear relationships
GBR/XGB → captures non-linear interactions

Advantages over RF:
Reduces systematic bias
Improves generalization
Balances underfitting and overfitting
Incorporates model diversity

Result: Lowest bias and more stable predictions across years

Final Prediction Setup
Model: EML
Resolution: 0.25°
Parameters: All
Input Month: January (stable & comparable to March)
