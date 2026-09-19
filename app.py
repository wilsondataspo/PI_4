from flask import Flask, render_template, request, jsonify, abort
import pandas as pd
import json
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from model_utils import predict, load_model

app = Flask(_name_)

# ------------------ Carregamento único dos dados ------------------
DATA_PATH = 'gym_membership.csv'
METRICS_PATH = 'model_metrics.json'
X_TEST_PATH = 'X_test.csv'
Y_TEST_PATH = 'y_test.csv'

df = pd.read_csv(DATA_PATH)

# Métricas (se existirem)
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH) as f:
        metrics = json.load(f)


# ------------------ Rotas ------------------

@app.route('/')
def index():
    """Visão geral do dataset."""
    info = {
        'total_registros': int(df.shape[0]),
        'taxa_renovacao': float(df['renewed'].mean() * 100),
        'num_features': int(df.shape[1] - 1),
        'colunas': list(df.columns),
        'tipos': df.dtypes.astype(str).to_dict(),
        'amostra': df.head(10).to_dict(orient='records')
    }
    return render_template('index.html', info=info, page='index')


@app.route('/eda')
def eda():
    """Análise exploratória: retorna dados agregados para os gráficos."""
    # Distribuição da variável alvo
    target_counts = df['renewed'].value_counts().to_dict()

    # Boxplots para variáveis numéricas vs renovação
    num_cols = ['age', 'membership_months', 'visits_per_week',
                'avg_time_spent', 'monthly_fee', 'pt_sessions']
    box_data = {}
    for col in num_cols:
        box_data[col] = {
            '0': df[df['renewed'] == 0][col].dropna().tolist(),
            '1': df[df['renewed'] == 1][col].dropna().tolist()
        }

    # Categóricos vs renovação
    cat_cols = ['gender', 'plan_type', 'fitness_level']
    cat_data = {}
    for col in cat_cols:
        cat_data[col] = df.groupby([col, 'renewed']).size().unstack(fill_value=0).to_dict()

    # Correlação
    corr = df.select_dtypes(include='number').corr().round(2)
    corr_data = {
        'labels': corr.columns.tolist(),
        'matrix': corr.values.tolist()
    }

    # Distribuição de variáveis numéricas (histogramas)
    hist_data = {}
    for col in num_cols:
        hist_data[col] = df[col].dropna().tolist()

    return render_template(
        'eda.html',
        page='eda',
        target_counts=target_counts,
        box_data=json.dumps(box_data),
        cat_data=json.dumps(cat_data),
        corr_data=json.dumps(corr_data),
        hist_data=json.dumps(hist_data),
        num_cols=num_cols,
        cat_cols=cat_cols
    )


@app.route('/models')
def models_page():
    """Página com métricas dos modelos treinados."""
    if not metrics:
        abort(500, "Métricas não encontradas. Rode train_model.py primeiro.")

    # Extrai resultados
    results = metrics['results']
    best_name = metrics['best_model']

    # Calcula matriz de confusão e curva ROC para o melhor modelo (se houver dados de teste)
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
        auc_score = roc_auc_score(y_test, y_proba)
        roc_data = {
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'auc': float(auc_score)
        }

    # Importância das features (se o modelo permitir)
    importances = None
    model = load_model()
    classifier = model.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        preprocessor = model.named_steps['preprocessor']
        cat_encoder = preprocessor.named_transformers_['cat']
        cat_names = cat_encoder.get_feature_names_out(['gender', 'plan_type', 'fitness_level'])
        num_names = ['age', 'membership_months', 'visits_per_week', 'avg_time_spent',
                     'monthly_fee', 'pt_sessions', 'engagement_score', 'cost_per_visit']
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
        cm_data=json.dumps(cm_data),
        roc_data=json.dumps(roc_data),
        importances=json.dumps(importances)
    )


@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Formulário de previsão."""
    if request.method == 'POST':
        try:
            # Recebe os dados do formulário
            data = {
                'age': float(request.form['age']),
                'gender': request.form['gender'],
                'membership_months': float(request.form['membership_months']),
                'visits_per_week': float(request.form['visits_per_week']),
                'avg_time_spent': float(request.form['avg_time_spent']),
                'fitness_level': int(request.form['fitness_level']),
                'plan_type': request.form['plan_type'],
                'monthly_fee': float(request.form['monthly_fee']),
                'attended_group_classes': int(request.form['attended_group_classes']),
                'pt_sessions': int(request.form['pt_sessions']),
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


# ------------------ Execução ------------------
if _name_ == '_main_':
    app.run(debug=True, host='0.0.0.0', port=5000)