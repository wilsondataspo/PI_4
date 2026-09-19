from flask import Flask, render_template, request, jsonify, abort
import pandas as pd
import json
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from model_utils import predict, load_model

app = Flask(__name__)

DATA_PATH     = 'gym_membership.csv'
METRICS_PATH  = 'model_metrics.json'
X_TEST_PATH   = 'X_test.csv'
Y_TEST_PATH   = 'y_test.csv'
TARGET        = 'renewed_membership'

df = pd.read_csv(DATA_PATH)

metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)


@app.route('/')
def index():
    info = {
        'total_registros': int(df.shape[0]),
        'taxa_renovacao':  float(df[TARGET].mean() * 100),
        'num_features':    int(df.shape[1] - 2),  # exclui id e alvo
        'colunas':         list(df.columns),
        'tipos':           df.dtypes.astype(str).to_dict(),
        'amostra':         df.head(10).to_dict(orient='records')
    }

    return render_template('index.html', info=info, page='index')


@app.route('/eda')
def eda():
    target_counts = df[TARGET].value_counts().to_dict()

    num_cols = ['age', 'subscription_length', 'num_logins', 'num_complaints',
                'num_classes_attended', 'avg_session_time']

    box_data = {}
    for col in num_cols:
        box_data[col] = {
            '0': df[df[TARGET] == 0][col].dropna().tolist(),
            '1': df[df[TARGET] == 1][col].dropna().tolist()
        }

    cat_cols = ['gender', 'membership_type']
    cat_data = {}
    for col in cat_cols:
        cat_data[col] = df.groupby([col, TARGET]).size().unstack(fill_value=0).to_dict()

    # Correlação (só numéricas)
    corr = df.select_dtypes(include='number').corr().round(2)
    corr_data = {'labels': corr.columns.tolist(), 'matrix': corr.values.tolist()}

    return render_template(
        'eda.html',
        page='eda',
        target_counts=target_counts,
        box_data=json.dumps(box_data),
        cat_data=json.dumps(cat_data),
        corr_data=json.dumps(corr_data),
        num_cols=num_cols,
        cat_cols=cat_cols
    )


@app.route('/models')
def models_page():
    if not metrics:
        abort(500, "Métricas não encontradas. Rode train_model.py primeiro.")

    results   = metrics['results']
    best_name = metrics['best_model']

    cm_data = None
    roc_data = None
    if os.path.exists(X_TEST_PATH) and os.path.exists(Y_TEST_PATH):
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH).squeeze()

        model = load_model()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cm = confusion_matrix(y_test, y_pred).tolist()
        cm_data = {'matrix': cm, 'labels': ['Não Renovou', 'Renovou']}

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(roc_auc_score(y_test, y_proba))
        }

    # Importância das features
    importances = None
    model = load_model()
    classifier = model.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        preprocessor = model.named_steps['preprocessor']
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_names = cat_encoder.get_feature_names_out(['gender', 'membership_type'])
        num_names = ['age', 'subscription_length', 'num_logins', 'num_complaints',
                     'num_classes_attended', 'avg_session_time',
                     'engagement_score', 'logins_per_month', 'classes_per_month',
                     'complaint_ratio', 'has_complaints']
        feature_names = num_names + list(cat_names)
        importances = sorted(
            zip(feature_names, classifier.feature_importances_.tolist()),
            key=lambda x: x[1], reverse=True
        )[:15]

    return render_template(
        'models.html',
        page='models',
        results=results,
        best_name=best_name,
        cm_data=cm_data,
        roc_data=roc_data,
        importances=importances
        # cm_data=json.dumps(cm_data),
        # roc_data=json.dumps(roc_data),
        # importances=json.dumps(importances)
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        try:
            data = {
                'age':                   float(request.form['age']),
                'gender':                request.form['gender'],
                'subscription_length':   float(request.form['subscription_length']),
                'membership_type':       request.form['membership_type'],
                'num_logins':            float(request.form['num_logins']),
                'num_complaints':        float(request.form['num_complaints']),
                'num_classes_attended':  float(request.form['num_classes_attended']),
                'avg_session_time':      float(request.form['avg_session_time']),
            }
            pred, proba = predict(data)
            return jsonify({
                'success': True,
                'prediction': pred,
                'probability': proba,
                'message': 'Cliente com alta chance de renovar.' if pred == 1
                           else 'Cliente com baixa chance de renovar.'
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    return render_template('predict.html', page='predict')


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=5000)
    app.run(debug=True)