#%%
# Importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import shap

# Read the dataset
dataset_path = 'UnifiedBugDataset-1.0/BugPrediction/csv-class/PDE_UI-3.4.1-Unified.csv'
bug_predict = pd.read_csv(dataset_path)

# Drop columns with more than 25% missing values
threshold = len(bug_predict) * 0.25
bug_predict = bug_predict.loc[:, bug_predict.isnull().sum() <= threshold]

# Drop categorical columns
categorical_cols = bug_predict.select_dtypes(include=['object']).columns
bug_predict = bug_predict[[col for col in bug_predict.columns if col not in categorical_cols]]

# Convert the target column ('bug') to binary: 1 for bugs, 0 otherwise
bug_predict['bug'] = bug_predict['bug'].replace(
    [value for value in bug_predict['bug'].unique() if value != 0], 1
)

# Feature importance using Random Forest
X = bug_predict.drop('bug', axis=1)
y = bug_predict['bug']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_reduced = pd.DataFrame(X_scaled, columns=X.columns)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_reduced, y)

feature_importances = pd.DataFrame({
    'Feature': X_reduced.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot top 20 feature importances
plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'].head(20), feature_importances['Importance'].head(20), align='center')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances')
plt.show()

#%%
# SHAP explanation for Random Forest
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_reduced)

# SHAP summary plot
shap.summary_plot(shap_values[1], X_reduced, plot_type="bar")
shap.summary_plot(shap_values[1], X_reduced)

#%%

# Select top 20 features
top_features = feature_importances['Feature'][:20]
X_reduced = X_reduced[top_features]

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_reduced, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train base models (Random Forest and XGBoost)
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Meta-learner using Logistic Regression
meta_model = LogisticRegression()
meta_model.fit(
    pd.DataFrame({'RF': rf.predict_proba(X_train)[:, 1], 'XGB': xgb.predict_proba(X_train)[:, 1]}),
    y_train
)

# Make predictions on the test set
test_preds = meta_model.predict(
    pd.DataFrame({'RF': rf.predict_proba(X_test)[:, 1], 'XGB': xgb.predict_proba(X_test)[:, 1]})
)

# Evaluate the model
print('Classification Report:')
print(classification_report(y_test, test_preds, zero_division=0))

# Plot confusion matrix
cm = confusion_matrix(y_test, test_preds)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Testing Set)")
plt.show()
