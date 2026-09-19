import joblib
import pandas as pd

_model = None

def load_model(path='best_model.pkl'):
    global _model
    if _model is None:
        _model = joblib.load(path)
    return _model


def prepare_input(data: dict) -> pd.DataFrame:
    """Recebe dict do formulário e devolve DataFrame com features derivadas."""
    df = pd.DataFrame([data])

    # Conversões numéricas
    numeric = ['age', 'subscription_length', 'num_logins', 'num_complaints',
               'num_classes_attended', 'avg_session_time']
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Features derivadas (idênticas às do treino!)
    df['engagement_score']  = df['num_logins'] * df['avg_session_time']
    df['logins_per_month']  = df['num_logins'] / (df['subscription_length'] + 1e-6)
    df['classes_per_month'] = df['num_classes_attended'] / (df['subscription_length'] + 1e-6)
    df['complaint_ratio']   = df['num_complaints'] / (df['subscription_length'] + 1e-6)
    df['has_complaints']    = (df['num_complaints'] > 0).astype(int)

    return df


def predict(data: dict):
    model = load_model()
    df = prepare_input(data)
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0, 1])
    return pred, proba