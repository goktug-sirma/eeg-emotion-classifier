from .preprocess import preprocess_data
from .features import extract_features
from .models import train_and_evaluate

def run_pipeline(data_path):
    data = preprocess_data(data_path)
    X, y = extract_features(data)
    train_and_evaluate(X, y)
