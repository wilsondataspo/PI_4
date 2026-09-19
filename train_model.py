import pandas as pd
# import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ------------------ Carregar dados ------------------
df = pd.read_csv('gym_membership.csv')

# ------------------ Mapear códigos de membership_type ------------------
MEMBERSHIP_MAP = {
    1: 'Basic',
    3: 'Standard',
    6: 'Premium',
    12: 'Premium',
}
df['membership_type'] = df['membership_type'].map(MEMBERSHIP_MAP)

# Confere se sobrou algum valor não mapeado (NaN)
assert df['membership_type'].isna().sum() == 0, \
    f"Valores não mapeados em membership_type: {df['membership_type'].unique()}"

# Descartar ID
df = df.drop(columns=['customer_id'])

# ------------------ Engenharia de features ------------------
df['engagement_score']   = df['num_logins'] * df['avg_session_time']
df['logins_per_month']   = df['num_logins'] / (df['subscription_length'] + 1e-6)
df['classes_per_month']  = df['num_classes_attended'] / (df['subscription_length'] + 1e-6)
df['complaint_ratio']    = df['num_complaints'] / (df['subscription_length'] + 1e-6)
df['has_complaints']     = (df['num_complaints'] > 0).astype(int)

# ------------------ Alvo ------------------
TARGET = 'renewed_membership'
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ------------------ Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------ Colunas ------------------
numeric_features = [
    'age', 'subscription_length', 'num_logins', 'num_complaints',
    'num_classes_attended', 'avg_session_time',
    'engagement_score', 'logins_per_month', 'classes_per_month',
    'complaint_ratio', 'has_complaints'
]
categorical_features = ['gender', 'membership_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# ------------------ Balanceamento ------------------
use_smote = y_train.value_counts(normalize=True).min() < 0.4
sampler = SMOTE(random_state=42) if use_smote else 'passthrough'
print(f"SMOTE ativado: {use_smote}")

# ------------------ Modelos ------------------
# models = {
#     'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
#     'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
#     'Gradient Boosting': GradientBoostingClassifier(random_state=42),
#     'XGBoost': xgb.XGBClassifier(eval_metric='logloss', random_state=42)
# }

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(
        eval_metric='logloss', random_state=42,
        scale_pos_weight=62/38)
}

results = {}
best_auc = 0
best_pipeline = None
best_name = None

for name, model in models.items():
    pipe = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('sampler', sampler),
        ('classifier', model)
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='roc_auc')

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)

    results[name] = {
        'cv_auc_mean': float(cv_scores.mean()),
        'cv_auc_std':  float(cv_scores.std()),
        'test_auc':    float(test_auc),
        'accuracy':    float(acc),
    }
    print(f"{name}: CV AUC={cv_scores.mean():.4f}, Test AUC={test_auc:.4f}, Acc={acc:.4f}")

    if test_auc > best_auc:
        best_auc = test_auc
        best_pipeline = pipe
        best_name = name

print(f"\nMelhor modelo: {best_name} (AUC={best_auc:.4f})")

# ------------------ Salvar artefatos ------------------
joblib.dump(best_pipeline, 'best_model.pkl')

with open('model_metrics.json', 'w') as f:
    json.dump({'best_model': best_name, 'results': results}, f, indent=2)

X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

print("Artefatos salvos: best_model.pkl, model_metrics.json, X_test.csv, y_test.csv")