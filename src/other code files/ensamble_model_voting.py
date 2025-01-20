# Import necessary libraries
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import joblib

# Load and preprocess data
dataset_path = 'UnifiedBugDataset-1.0/BugPrediction/csv-class/PDE_UI-3.4.1-Unified.csv'
bug_predict = pd.read_csv(dataset_path)

# Drop columns with more than 25% missing values
threshold = len(bug_predict) * 0.25
bug_predict = bug_predict.loc[:, (bug_predict.isnull().sum() <= threshold) | (bug_predict.columns == 'bug')]

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

# Select top 20 features using feature importances from RandomForestClassifier
rf_feature_importance = GradientBoostingClassifier(random_state=42)
rf_feature_importance.fit(X_scaled, y)

# Get top 20 features
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_feature_importance.feature_importances_
}).sort_values(by='Importance', ascending=False)

top_features = feature_importances['Feature'].iloc[:20]
X_top_features = pd.DataFrame(X_scaled, columns=X.columns)[top_features]

# Plot Feature Importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'][:20], feature_importances['Importance'][:20], align='center')
plt.gca().invert_yaxis()
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.show()

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_top_features, y)

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# File path to save the best TabNet model
tabnet_model_path = "best_tabnet_model.pkl"
#%%
# Optuna Optimization for TabNet
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

    # Initialize TabNet model
    model = TabNetClassifier(**params)

    # Fit the model
    model.fit(
        X_train.values, y_train.values,  # Convert to NumPy arrays
        eval_set=[(X_val.values, y_val.values)],  # Convert to NumPy arrays
        eval_metric=['accuracy'],
        patience=50,
        max_epochs=200
    )

    # Predict on validation set
    preds = model.predict(X_val.values)

    # Calculate accuracy for Optuna
    return classification_report(y_val, preds, output_dict=True)['accuracy']

# Perform Optuna Optimization
study_tabnet = optuna.create_study(direction='maximize')
study_tabnet.optimize(objective_tabnet, n_trials=20)

# Save the best TabNet model after optimization
best_params = study_tabnet.best_params
best_tabnet_model = TabNetClassifier(**best_params)
best_tabnet_model.fit(
    X_train.values, y_train.values,
    eval_set=[(X_val.values, y_val.values)],
    eval_metric=['accuracy'],
    patience=50,
    max_epochs=200
)
# Save the trained model
joblib.dump(best_tabnet_model, "best_tabnet_model.pkl")
#%%
# Load the best model for subsequent work
# Load the best TabNet model for subsequent work
loaded_tabnet_model = joblib.load("best_tabnet_model.pkl")

# Meta-Learner (Random Forest, Gradient Boosting, XGBoost)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Train Random Forest, Gradient Boosting, and XGBoost
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Meta-learner training
meta_features_train = pd.DataFrame({
    'RF': rf.predict_proba(X_train)[:, 1],
    'GB': gb.predict_proba(X_train)[:, 1],
    'XGB': xgb.predict_proba(X_train)[:, 1]
})
meta_model = LogisticRegression(max_iter=1000, random_state=42)
meta_model.fit(meta_features_train, y_train)

# Meta-learner predictions (probabilities)
meta_features_test = pd.DataFrame({
    'RF': rf.predict_proba(X_test)[:, 1],
    'GB': gb.predict_proba(X_test)[:, 1],
    'XGB': xgb.predict_proba(X_test)[:, 1]
})
meta_probs = meta_model.predict_proba(meta_features_test)[:, 1]

# TabNet predictions (probabilities)
tabnet_probs = loaded_tabnet_model.predict_proba(X_test.values)[:, 1]

# Assign weights
meta_weight = 100  # Higher weight for meta_model
tabnet_weight = 1  # Lower weight for TabNet model

# Combine probabilities with weights
final_probs = (meta_probs * meta_weight + tabnet_probs * tabnet_weight) / (meta_weight + tabnet_weight)

# Final predictions based on weighted probabilities
final_preds = [1 if prob > 0.5 else 0 for prob in final_probs]

# Evaluate the final ensemble model
print("Classification Report (Final Ensemble):")
print(classification_report(y_test, final_preds, zero_division=0))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Final Ensemble)")
plt.tight_layout()
plt.savefig("final_ensemble_confusion_matrix.png")
plt.show()

#%%
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC 

# Load the TabNet model
loaded_tabnet_model = joblib.load("best_tabnet_model.pkl")
tabnet_probs = loaded_tabnet_model.predict_proba(X_test.values)[:, 1]
tabnet_preds = loaded_tabnet_model.predict(X_test.values)

# Train individual models
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
log_reg = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(probability=True, random_state=42)

# Fit models
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
xgb.fit(X_train, y_train)
log_reg.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Get probabilities from each model
rf_probs = rf.predict_proba(X_test)[:, 1]
gb_probs = gb.predict_proba(X_test)[:, 1]
xgb_probs = xgb.predict_proba(X_test)[:, 1]
log_reg_probs = log_reg.predict_proba(X_test)[:, 1]
svm_probs = svm.predict_proba(X_test)[:, 1]

# Combine predictions using weighted probabilities
# rf_weight = 10
# gb_weight = 10
# xgb_weight = 10
log_reg_weight = 1
svm_weight = 1
tabnet_weight = 4

# Weighted average of probabilities
final_probs = (
    # (rf_probs * rf_weight) +
    # (gb_probs * gb_weight) +
    # (xgb_probs * xgb_weight) +
    (log_reg_probs * log_reg_weight) +
    (svm_probs * svm_weight) +
    (tabnet_probs * tabnet_weight)
) / (log_reg_weight + svm_weight + tabnet_weight)

# Final predictions based on combined probabilities
final_preds = [1 if prob > 0.5 else 0 for prob in final_probs]

# Evaluate the final ensemble model
print("Classification Report (Final Ensemble):")
print(classification_report(y_test, final_preds, zero_division=0))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Final Ensemble)")
plt.tight_layout()
plt.savefig("final_ensemble_confusion_matrix.png")
plt.show()

#%%