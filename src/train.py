import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.under_sampling import RandomUnderSampler

import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
DATA = "Financial Distress.csv"
MODEL_DIR = "artifacts"

os.makedirs(MODEL_DIR, exist_ok=True)

""" Dataset loading & prepration ............................................................................................
"""

def load_data():

    df = pd.read_csv(f"{MODEL_DIR}/{DATA}")
    df['Financial Distress'] = (df['Financial Distress'] < -0.5).astype(int) # Binarise target
    print(f"Dataset loaded with shape: {df.shape}")
    
    return df

""" Data preprocessing ............................................................................................
"""
def split_data(df):

    X = df.drop('Financial Distress', axis=1)
    y = df['Financial Distress']

    # 70 / 30 first split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)

    # 15 / 15 second split
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)
    
    # Save test set for later use as new dataset for evaluation
    test_set = pd.concat([X_test, y_test], axis=1)
    test_set.to_csv(os.path.join(MODEL_DIR, 'test_set.csv'), index=False)

    print(f"Data is splitted in Train, Val and Test sets")
    
    return X_train, X_val, y_train, y_val

""" Model utilities  ............................................................................................
"""
def evaluate_model(name, model, X_tr, y_tr, X_ev, y_ev,
                   sampler=None, sampler_name='None'):
    """Fit model (optionally with sampler) and return metrics dict."""
    if sampler is not None:
        X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_ev)
    return {
        'Model'        : name,
        'Sampling'     : sampler_name,
        'Recall'       : round(recall_score(y_ev, y_pred, pos_label=1), 4),
        'F1'           : round(f1_score(y_ev, y_pred, pos_label=1), 4),
        '_y_pred'      : y_pred,   # keep for confusion matrix
    }


# Model factory — fresh instance each call
def get_models():
    return [
        ('Random Forest',       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ('XGBoost',             XGBClassifier(eval_metric='logloss', random_state=RANDOM_STATE)),
    ]



""" Random Under-Sampling ............................................................................................
"""

def random_under_sampling(X_train, y_train):
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_rus, y_rus = rus.fit_resample(X_train, y_train)
    return X_rus, y_rus

""" Training Best Models ............................................................................................
"""
def train_models(X_train, y_train, X_val, y_val):
    
    for name, model in get_models():
        print(f"\n{name} - Training with Random Under-Sampling...")
        # Apply undersampling to training data only — mirrors the notebook
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
        # Train on resampled, evaluate on original val set
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_val)
        
        recall = round(recall_score(y_val, y_pred, pos_label=1), 4)
        f1     = round(f1_score(y_val, y_pred, pos_label=1), 4)
        
        print(f'Recall={recall:.4f}  F1={f1:.4f}')
        
        with open(os.path.join(MODEL_DIR, f'{name.replace(" ", "_")}.pkl'), 'wb') as f:
            joblib.dump(model, f)
    
    print("\nAll models trained and saved.")



if __name__ == "__main__":
    df = load_data()
    X_train, X_val, y_train, y_val = split_data(df)
    train_models(X_train, y_train, X_val, y_val)





