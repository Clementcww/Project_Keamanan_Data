import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, roc_auc_score, confusion_matrix, roc_curve, classification_report
from sklearn.pipeline import Pipeline
import os

# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('data/plots'):
    os.makedirs('data/plots')

def load_data(filepath):
    df = pd.read_csv(filepath)
    # Ensure no missing values in text
    df = df.dropna(subset=['text'])
    return df

def train_and_evaluate(df):
    X = df['text']
    y = df['label_num'] # Assuming label_num is 0/1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    best_model = None
    best_f1 = 0
    best_model_name = ""
    
    results = []

    print("Training and Evaluating Models...")
    
    for name, model in models.items():
        print(f"\nProcessing {name}...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english')),
            ('clf', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC AUC': roc_auc
        })
        
        # Save Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'data/plots/confusion_matrix_{name.replace(" ", "_")}.png')
        plt.close()
        
        # Save ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure(figsize=(6, 4))
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.savefig(f'data/plots/roc_curve_{name.replace(" ", "_")}.png')
        plt.close()
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = pipeline
            best_model_name = name

    # Save Best Model
    print(f"\nBest Model: {best_model_name} with F1 Score: {best_f1:.4f}")
    joblib.dump(best_model, 'models/spam_classifier.joblib')
    print("Best model saved to 'models/spam_classifier.joblib'")
    
    # Save Results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/model_evaluation_results.csv', index=False)
    print("Evaluation results saved to 'data/model_evaluation_results.csv'")

if __name__ == "__main__":
    try:
        df = load_data('data/email.csv')
        train_and_evaluate(df)
    except Exception as e:
        print(f"Error: {e}")
