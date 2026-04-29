**ML4Monsoons – Monsoon Rainfall Prediction**

**Problem Statement**

The objective of this project is to predict monsoon rainfall (June–September) in advance using climate variables available during earlier months (January, February, March).

The goal is to estimate rainfall values as close as possible to IMD (Indian Meteorological Department) observations using multiple machine learning models and spatial resolutions.

**Aim**

- Identify the best-performing model
- Determine the optimal spatial resolution (0.25° vs 1°)
- Compare all-parameter vs single-parameter inputs
- Identify the best prediction month (Jan / Feb / Mar)

Final Objective: Build a reliable and accurate early rainfall prediction system.

**Why Predict from Jan–Mar?**

Monsoon rainfall is influenced by large-scale atmospheric patterns that begin months earlier.

- January–March contain early climate signals
- Enables long-range forecasting
- Supports:
     - Agriculture planning
     - Water resource management
     - Disaster preparedness

**Approach Overview**

**Data Used**
- Gridded climate dataset
- Two spatial resolutions:

       - 0.25° (high resolution)
       - 1° (coarse resolution)
**Features**
- Precipitation (PR)
- Geopotential Height (GZ)
- Temperature (TR)

**Target Variable**

IMD rainfall values

**Models Implemented**

The following models were trained and compared:

- KNN
- Random Forest (RF)
- Dense Model
- Ensemble DNN
- Optuna
- EML
- Stacked Ensemble 

**Hyperparameters**

**KNN**

- n_neighbors: 3, 5, 7, 9, 11
- weights: uniform / distance

**Ridge Regression**
- alpha = 1.0

**Random Forest (RF)**

- n_estimators: 50–300
- max_depth: 10–None
- min_samples_split: 2, 5, 10
- min_samples_leaf: 1, 2, 4
- max_features: auto, sqrt, log2

**Dense Model**

- Architecture: 256 → 128 → 64 → 32
- Activation: ReLU
- Dropout: 0.2
- L2 Regularization: 0.0005
- Optimizer: Adam
- Learning Rate: 1e-4

**Ensemble DNN**

- Same architecture as Dense model
- Multiple seeds: [1, 42, 101, 202, 303, 404, 505]
- Final prediction = average of all models

**Optuna Model**

- Units per layer: 16–512 (tuned)
- Dropout: 0.1–0.4
- L2 Regularization: 1e-5 → 1e-3
- Learning Rate: 1e-5 → 1e-3

**EML**

- Base Models:
     - KNN
     - Ridge
     - Gradient Boosting
     
- Meta Model:

      - Linear Regression

**Stacked Ensemble**

Base Models:
- KNN
- Ridge
- XGBoost

Additional Features:
- Median prediction (Q50)
- Uncertainty band (Q90 − Q10)

Meta Model:
- Linear Regression / XGBoost

**Experiment Workflow**

**Step 1: Model Comparison (1° Resolution)**

- Trained all models on Jan, Feb, Mar
- Best Model: Optuna

**Step 2: Parameter Comparison (1°)**

- All parameters vs single parameter
- Best: All parameters

**Step 3: Month Comparison (1°)**

- Jan vs Feb vs Mar
- Best Month: March

**Step 4: Model Comparison (0.25° Resolution)**

- Trained all models
- Best Model: Random Forest (RF)

**Step 5: Parameter Comparison (0.25°)**

- All parameters vs single parameter
- Best: All parameters

**Step 6: Month Comparison (0.25°)**

- Best Month: March

**Step 7: Resolution Comparison**

- 0.25° vs 1°
- Best Resolution: 0.25° (higher spatial accuracy)

**Important Observation**

Using:

- Random Forest
- 0.25° resolution
- All parameters
- Jan dataset 

Result: Predictions showed consistent underestimation.

**Final Model Decision**

All models were tested on unseen data (2019–2023):

**EML achieved the lowest bias.**

**Why EML Performed Better than RF?**

EML combines strengths of multiple models:

- KNN captures local patterns
- Ridge models linear relationships
- GBR/XGBoost captures non-linear interactions

**Advantages over RF**

- Reduces systematic bias
- Improves generalization
- Balances underfitting and overfitting
- Leverages model diversity

Result: More stable and accurate predictions across years.

**Final Prediction Setup**
- **Model**: EML
- **Resolution**: 0.25°
- **Parameters**: All
- **Input Month**: January (chosen for stability and comparable performance with March)

**Repository Structure**

ML4Monsoons/

├── README.md


├── data/

│   ├── jan/

│   ├── feb/

│   └── mar/


├── models/

│   ├── jan/

│   │   ├── 0.25/

│   │   │   ├── all_parameters/

│   │   │   └── single_parameter/

│   │   └── 1_degree/

│   │       ├── all_parameters/

│   │       └── single_parameter/

│   │

│   ├── feb/

│   └── mar/



**Conclusion**
- Best Resolution: 0.25°
- Best Model: EML
- Best Input: All parameters
- Best Month: March

Final system provides accurate and low-bias rainfall prediction.
