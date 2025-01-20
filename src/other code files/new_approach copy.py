#%%
# Importing the required libraries
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
import shap
import optuna
import torch
#%%
# Read the dataset
dataset_path = 'UnifiedBugDataset-1.0/BugPrediction/csv-class/PDE_UI-3.4.1-Unified.csv'
bug_predict = pd.read_csv(dataset_path)
#%%
# Drop columns with more than 25% missing values
threshold = len(bug_predict) * 0.25
bug_predict = bug_predict.loc[:, (bug_predict.isnull().sum() <= threshold) | (bug_predict.columns == 'bug')]
#%%
# Drop categorical columns, but keep the target column 'bug'
categorical_cols = bug_predict.select_dtypes(include=['object']).columns
bug_predict = bug_predict[[col for col in bug_predict.columns if col not in categorical_cols or col == 'bug']]

# Convert the target column ('bug') to binary: 1 for bugs, 0 otherwise
bug_predict['bug'] = bug_predict['bug'].replace(
    [value for value in bug_predict['bug'].unique() if value != 0], 1
)

# Separate features and target
X = bug_predict.drop('bug', axis=1)
y = bug_predict['bug']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dimensionality reduction using PCA
pca = PCA(n_components=30)
X_pca = pca.fit_transform(X_scaled)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_pca, y)

# Train-validation-test split (60-20-20)
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Optuna Optimization for TabNet
def objective_tabnet(trial):
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'momentum': trial.suggest_float('momentum', 0.01, 0.4),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 0.0001, 0.01),
    }
    model = TabNetClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['accuracy'],
        patience=50,
        max_epochs=200
    )
    preds = model.predict(X_val)
    return classification_report(y_val, preds, output_dict=True)['accuracy']

study_tabnet = optuna.create_study(direction='maximize')
study_tabnet.optimize(objective_tabnet, n_trials=20)
best_tabnet_params = study_tabnet.best_params

# Train optimized TabNet
tabnet_model = TabNetClassifier(**best_tabnet_params)
tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=['accuracy'],
    patience=50,
    max_epochs=200
)

# Evaluate the model on the test set
test_preds = tabnet_model.predict(X_test)

# Classification report
print('Classification Report (Test Set):')
print(classification_report(y_test, test_preds, zero_division=0))
#%%


#%%
