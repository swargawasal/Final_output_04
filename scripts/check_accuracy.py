"""
Check Watermark Model Accuracy
------------------------------
Calculates accuracy, precision, recall, and F1 score of the current watermark model
against the collected dataset.
"""

import os
import sys
import pickle
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = "models"
DATASET_FILE = os.path.join(MODEL_DIR, "watermark_dataset.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "watermark_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "watermark_scaler.pkl")

def check_accuracy():
    if not os.path.exists(DATASET_FILE):
        print(json.dumps({"error": "Dataset not found", "path": DATASET_FILE}))
        return

    try:
        # Load Data
        df = pd.read_csv(DATASET_FILE)
        print(f"📊 Total Samples: {len(df)}")
        
        if len(df) < 10:
            print("⚠️ Not enough data to calculate meaningful accuracy.")
            return

        X = df.drop(columns=['label'])
        y = df['label']

        # Load Model
        if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
            print("⚠️ Model or Scaler not found. Cannot evaluate current model.")
            print("💡 Tip: Run 'python scripts/nightly_retrain.py' to train a model.")
            return

        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)

        # Prepare Data
        X_scaled = scaler.transform(X)

        # Predict
        y_pred = model.predict(X_scaled)

        # Metrics
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)

        results = {
            "accuracy": f"{acc:.2%}",
            "precision": f"{prec:.2%}",
            "recall": f"{rec:.2%}",
            "f1_score": f"{f1:.2%}",
            "confusion_matrix": {
                "true_negatives": int(cm[0][0]) if len(cm) > 1 else 0,
                "false_positives": int(cm[0][1]) if len(cm) > 1 else 0,
                "false_negatives": int(cm[1][0]) if len(cm) > 1 else 0,
                "true_positives": int(cm[1][1]) if len(cm) > 1 else 0
            }
        }

        print(json.dumps(results, indent=2))
        
        # Interpretation
        print("\n--- Interpretation ---")
        print(f"✅ Accuracy: {acc:.2%} (Overall correctness)")
        print(f"🎯 Precision: {prec:.2%} (When it says 'Watermark', how often is it right?)")
        print(f"🔍 Recall: {rec:.2%} (How many real watermarks did it find?)")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_accuracy()
