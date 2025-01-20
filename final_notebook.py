#%%
#%% Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
import joblib

# Load the dataset
dataset_path = 'UnifiedBugDataset-1.0/BugPrediction/csv-class/PDE_UI-3.4.1-Unified.csv'
bug_predict = pd.read_csv(dataset_path)

# Drop columns with more than 25% missing values
threshold = len(bug_predict) * 0.25
bug_predict = bug_predict.loc[:, (bug_predict.isnull().sum() <= threshold) | (bug_predict.columns == 'bug')]

# Drop categorical columns, but keep the target column 'bug'
categorical_cols = bug_predict.select_dtypes(include=['object']).columns
bug_predict = bug_predict[[col for col in bug_predict.columns if col not in categorical_cols or col == 'bug']]
bug_predict['bug'] = bug_predict['bug'].replace(
    [value for value in bug_predict['bug'].unique() if value != 0], 1
)

#Exploratory Analysis

# Basic statistics of numerical features
print("\nBasic Statistics:")
print(bug_predict.describe())

# Plot target class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='bug', data=bug_predict, palette='coolwarm')
plt.title('Bug Class Distribution')
plt.xlabel('Bug (0 = No Bug, 1 = Bug)')
plt.ylabel('Count')
plt.savefig('bug_class_distribution.png')
plt.show()

# Plot correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = bug_predict.drop(columns="bug").corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.savefig('correlation_heatmap.png')
plt.show()
#%%
# Feature importance using correlation with the target
correlation_with_target = correlation_matrix['bug'].drop('bug')
correlation_with_target = correlation_with_target.abs().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
correlation_with_target.plot(kind='bar', color='skyblue')
plt.title('Feature Correlation with Target (Bug)')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.xticks(rotation=90)
plt.savefig('feature_correlation_with_target.png')
plt.show()

# Pairplot to analyze relationships between features
sampled_data = bug_predict.sample(frac=0.1, random_state=42)  # Use sample to avoid memory issues
sns.pairplot(sampled_data, hue='bug', diag_kind='kde', palette='husl')
plt.savefig('pairplot_features.png')
plt.show()

# Box plots to detect outliers
plt.figure(figsize=(14, 6))
for i, col in enumerate(bug_predict.columns[:6]):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(y=col, data=bug_predict, palette='coolwarm')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.savefig('boxplots_features.png')
plt.show()

# Separate features and target
X = bug_predict.drop('bug', axis=1)
y = bug_predict['bug']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# Plot the distribution of target variable after SMOTE
plt.figure(figsize=(6, 4))
sns.countplot(x=y_resampled, palette='coolwarm')
plt.title('Bug Class Distribution After SMOTE')
plt.xlabel('Bug (0 = No Bug, 1 = Bug)')
plt.ylabel('Count')
plt.savefig('bug_class_distribution_smote.png')
plt.show()

#%%

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize models
models = {
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

# Feature importance and training for each model
from sklearn.inspection import permutation_importance

# Feature importance and training for each model
selected_features = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    
    if hasattr(model, 'feature_importances_'):
        # Tree-based models: Compute feature importances
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

    elif model_name == 'Logistic Regression':
        # Logistic Regression: Use absolute coefficients as feature importance
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.abs(model.coef_[0])
        }).sort_values(by='Importance', ascending=False)

    elif model_name == 'SVM':
        # SVM: Use Permutation Importance to compute feature importance
        perm_importance = permutation_importance(model, X_train, y_train, scoring='accuracy', n_repeats=10, random_state=42)
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance.importances_mean
        }).sort_values(by='Importance', ascending=False)

    else:
        # For unsupported models, skip feature importance
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': [0] * X.shape[1]
        })

    # Select top 20 features
    top_features = feature_importances['Feature'].head(20)
    selected_features[model_name] = top_features

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances['Feature'].head(20), feature_importances['Importance'].head(20), align='center')
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top 20 Feature Importances - {model_name}')
    plt.tight_layout()
    plt.savefig(f'feature_importances_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

#%%
#%%
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
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['accuracy'],
        patience=50,
        max_epochs=200
    )

    # Predict on validation set
    preds = model.predict(X_val)
    return classification_report(y_val, preds, output_dict=True)['accuracy']

# Perform Optuna Optimization
study_tabnet = optuna.create_study(direction='maximize')
study_tabnet.optimize(objective_tabnet, n_trials=20)

# Save the best TabNet model after optimization
best_params = study_tabnet.best_params
tabnet_model = TabNetClassifier(**best_params)
tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric=['accuracy'],
    patience=50,
    max_epochs=200
)

# Save the optimized TabNet model
joblib.dump(tabnet_model, "optimized_tabnet_model.pkl")
#%%
# Load the optimized TabNet model
optimized_tabnet_model = joblib.load("optimized_tabnet_model.pkl")
models['TabNet'] = joblib.load("optimized_tabnet_model.pkl")
# TabNet feature importance
tabnet_explain = optimized_tabnet_model.explain(X_train)

# Access the first element of the tuple for feature importance masks
masks = tabnet_explain[0]

# Compute feature importances
tabnet_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.mean(masks, axis=0)
}).sort_values(by='Importance', ascending=False)

# Select top 20 TabNet features
selected_features['TabNet'] = tabnet_importances['Feature'].head(20)

# Plot TabNet feature importances
plt.figure(figsize=(10, 6))
plt.barh(tabnet_importances['Feature'].head(20), tabnet_importances['Importance'].head(20), align='center')
plt.gca().invert_yaxis()
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances - TabNet')
plt.tight_layout()
plt.savefig('feature_importances_tabnet.png')
plt.show()
#%%
# Train TabNet with the full feature set (original input dimensions)
X_train_tabnet = pd.DataFrame(X_train, columns=X.columns)
X_test_tabnet = pd.DataFrame(X_test, columns=X.columns)

# Initialize a dictionary to store final results
final_results = {}

# Evaluate all models, including TabNet
for model_name, model in models.items():
    # Use selected features for non-TabNet models
    if model_name != "TabNet":
        X_train_selected = pd.DataFrame(X_train, columns=X.columns)[selected_features[model_name]]
        X_test_selected = pd.DataFrame(X_test, columns=X.columns)[selected_features[model_name]]

        # Train the model and make predictions
        model.fit(X_train_selected, y_train)
        preds = model.predict(X_test_selected)
    else:
        # For TabNet, use the full feature set
        preds = optimized_tabnet_model.predict(X_test_tabnet.values)

        # TabNet feature importance
        tabnet_explain = optimized_tabnet_model.explain(X_train_tabnet.values)
        masks = tabnet_explain[0]  # Access the first element of the tuple for feature importance masks

        # Compute feature importances for TabNet
        tabnet_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': np.mean(masks, axis=0)
        }).sort_values(by='Importance', ascending=False)

        # Plot TabNet feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(tabnet_importances['Feature'].head(20), tabnet_importances['Importance'].head(20), align='center')
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Top 20 Feature Importances - TabNet')
        plt.tight_layout()
        plt.savefig('feature_importances_tabnet.png')
        plt.show()

    # Generate the classification report
    report = classification_report(y_test, preds, output_dict=True)
    final_results[model_name] = report

    # Display classification report
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, preds, zero_division=0))

    # Display confusion matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.show()

# Create a summary of results for all models
results_summary = pd.DataFrame({
    'Model': list(final_results.keys()),
    'Accuracy': [final_results[model]['accuracy'] for model in final_results],
    'Precision': [final_results[model]['weighted avg']['precision'] for model in final_results],
    'Recall': [final_results[model]['weighted avg']['recall'] for model in final_results],
    'F1-Score': [final_results[model]['weighted avg']['f1-score'] for model in final_results]
})

# Display the results summary
print("\n=== Model Performance Summary ===")
print(results_summary)

# Save results summary to a CSV file
results_summary.to_csv("model_comparison_results_with_all_classifiers.csv", index=False)



#%%
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
# Function to evaluate models and plot ROC curves
def evaluate_model(model, X_test, y_test, model_name):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for positive class
    
    # Compute evaluation metrics
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    # Print evaluation metrics
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    
    # Return results dictionary
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC': auc
    }

# Initialize results storage
final_results = {}

# Evaluate each model
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    
    if model_name != "TabNet":
        X_test_selected = pd.DataFrame(X_test, columns=X.columns)[selected_features[model_name]]
    else:
        X_test_selected = X_test_tabnet.values  # Use full feature set for TabNet
    
    final_results[model_name] = evaluate_model(model, X_test_selected, y_test, model_name)

# # Evaluate Voting Classifier
# print("\nEvaluating Ensemble Voting Classifier...")
# final_results['Voting Classifier'] = evaluate_model(voting_clf, X_test, y_test, "Voting Classifier")

# # Evaluate Stacking Classifier
# print("\nEvaluating Stacking Classifier...")
# final_results['Stacking Classifier'] = evaluate_model(stacking_clf, X_test, y_test, "Stacking Classifier")

# Display all ROC curves
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line for reference
plt.title("ROC Curves for Different Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.savefig("roc_curves_all_models.png")
plt.show()

# Convert results to DataFrame and save
results_summary_df = pd.DataFrame.from_dict(final_results, orient='index')
print("\n=== Final Model Performance Summary ===")
print(results_summary_df)

# Save results to CSV file
results_summary_df.to_csv("model_performance_results.csv", index=True)

#%%

#%%
from sklearn.ensemble import VotingClassifier

from sklearn.base import BaseEstimator, ClassifierMixin

class TabNetWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, tabnet_model):
        self.tabnet_model = tabnet_model

    def fit(self, X, y):
        self.tabnet_model.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return self.tabnet_model.predict(X)

    def predict_proba(self, X):
        return self.tabnet_model.predict_proba(X)


models['TabNet'] = TabNetWrapper(tabnet_model)

# Voting Classifier with weighted models
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', models['XGBoost']),
        ('gb', models['Gradient Boosting']),
        ('rf', models['Random Forest']),
        ('lr', models['Logistic Regression']),
        ('svm', models['SVM']),
        ('tabnet', models['TabNet'])
    ],
    voting='soft',
    weights=[1,1,1,1,1,1]
)

# Train and evaluate voting classifier
voting_clf.fit(X_train, y_train)
voting_preds = voting_clf.predict(X_test)
voting_report = classification_report(y_test, voting_preds, output_dict=True)
print("\n=== Ensemble Voting Classifier ===")
print(classification_report(y_test, voting_preds, zero_division=0))

# Plot confusion matrix
cm = confusion_matrix(y_test, voting_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.title("Confusion Matrix - Ensemble Voting Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_voting_classifier.png")
plt.show()

# Save results
final_results = {'Ensemble Voting': voting_report}
results_summary = pd.DataFrame({
    'Model': list(final_results.keys()),
    'Accuracy': [final_results['Ensemble Voting']['accuracy']],
    'Precision': [final_results['Ensemble Voting']['weighted avg']['precision']],
    'Recall': [final_results['Ensemble Voting']['weighted avg']['recall']],
    'F1-Score': [final_results['Ensemble Voting']['weighted avg']['f1-score']]
})

print("\n=== Model Performance Summary ===")
print(results_summary)
results_summary.to_csv("voting_classifier_results.csv", index=False)


#%%%

from sklearn.ensemble import StackingClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
# Define base models
base_models = [
    ('xgb', models['XGBoost']),
    ('gb', models['Gradient Boosting']),
    ('rf', models['Random Forest']),
    ('lr', models['Logistic Regression']),
    ('svm', models['SVM']),
    #('tabnet', TabNetWrapper(tabnet_model))
]

# Define the meta-model
meta_model = TabNetWrapper(tabnet_model)

# Create the stacking classifier
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)

# Train and evaluate stacking classifier
stacking_clf.fit(X_train, y_train)
stacking_preds = stacking_clf.predict(X_test)
stacking_report = classification_report(y_test, stacking_preds, output_dict=True)
print("\n=== Stacking Classifier ===")
print(classification_report(y_test, stacking_preds, zero_division=0))

# Plot confusion matrix
cm = confusion_matrix(y_test, stacking_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Bug', 'Bug'], yticklabels=['No Bug', 'Bug'])
plt.title("Confusion Matrix - Stacking Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix_stacking_classifier.png")
plt.show()

# Save results
results_summary['Stacking'] = {
    'Accuracy': stacking_report['accuracy'],
    'Precision': stacking_report['weighted avg']['precision'],
    'Recall': stacking_report['weighted avg']['recall'],
    'F1-Score': stacking_report['weighted avg']['f1-score']
}



#%%


#%%