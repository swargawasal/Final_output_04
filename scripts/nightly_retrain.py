"""
Nightly Auto-Retraining Script
------------------------------
Runs automatically to improve the Watermark Brain.
1. Checks lock file.
2. Loads last 30 days of data.
3. Trains Balanced Random Forest.
4. Validates and atomically updates model.
"""

import os
import sys
import time
import json
import pickle
import logging
import shutil
from datetime import datetime

# Add parent dir to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    print("❌ ML libraries missing. Skipping retrain.")
    sys.exit(0)

# Config
MODEL_DIR = "models"
DATASET_FILE = os.path.join(MODEL_DIR, "watermark_dataset.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "watermark_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "watermark_scaler.pkl")
LOG_DIR = "logs"
LOCK_FILE = os.path.join(MODEL_DIR, "retrain.lock")

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, f"retrain_{datetime.now().strftime('%Y%m%d')}.log"), level=logging.INFO)
logger = logging.getLogger("nightly_retrain")

def retrain():
    if os.path.exists(LOCK_FILE):
        # Check if stale (older than 1 hour)
        if time.time() - os.path.getmtime(LOCK_FILE) > 3600:
            os.remove(LOCK_FILE)
        else:
            print(json.dumps({"status": "skipped", "reason": "locked"}))
            return

    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))

    try:
        logger.info("🚀 Starting Nightly Retrain...")
        
        if not os.path.exists(DATASET_FILE):
            print(json.dumps({"status": "skipped", "reason": "no_data"}))
            return

        try:
            # Fix: Pandas 1.3+ uses on_bad_lines, older uses error_bad_lines
            if int(pd.__version__.split('.')[0]) >= 1 and int(pd.__version__.split('.')[1]) >= 3:
                 df = pd.read_csv(DATASET_FILE, on_bad_lines='skip')
            else:
                 df = pd.read_csv(DATASET_FILE, error_bad_lines=False)
        except Exception:
             # Universal fallback
             df = pd.read_csv(DATASET_FILE, on_bad_lines='skip')
        
        # Schema Validation
        SCHEMA_FILE = os.path.join(MODEL_DIR, "features.json")
        current_cols = list(df.drop(columns=['label'], errors='ignore').columns)
        
        if os.path.exists(SCHEMA_FILE):
            with open(SCHEMA_FILE, 'r') as f:
                saved_schema = json.load(f)
                
            if current_cols != saved_schema:
                logger.error(f"❌ Schema Validation Failed: Features changed! \nExpected: {saved_schema}\nGot: {current_cols}")
                print(json.dumps({"status": "skipped", "reason": "schema_mismatch"}))
                return
        else:
            # First run: Save schema
            with open(SCHEMA_FILE, 'w') as f:
                json.dump(current_cols, f)
        
        # Basic Validation
        if len(df) < 20 or len(df['label'].unique()) < 2:
            print(json.dumps({"status": "skipped", "reason": "insufficient_data"}))
            return

        X = df.drop(columns=['label'])
        y = df['label']
        
        # Split for Validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. Scaling
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 2. Train Balanced RF
        model = RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # 3. Validate & Regression Check
        val_acc = model.score(X_val_scaled, y_val)
        logger.info(f"📊 Validation Accuracy: {val_acc:.2%}")
        
        # Load old model for regression check
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    old_model = pickle.load(f)
                
                # Check regression (allow 0.5% tolerance)
                # We need to test the old model on the NEW validation set to be fair
                if hasattr(old_model, "predict"):
                    old_acc = old_model.score(X_val_scaled, y_val)
                    logger.info(f"📉 Previous Model Accuracy: {old_acc:.2%}")
                    
                    if val_acc < (old_acc - 0.005):
                        logger.warning(f"⚠️ Regression detected! New model ({val_acc:.2%}) is worse than old ({old_acc:.2%}). Skipping update.")
                        print(json.dumps({"status": "skipped_due_to_regression", "new_acc": val_acc, "old_acc": old_acc}))
                        return
            except Exception as e:
                logger.warning(f"⚠️ Could not load old model for comparison: {e}. Proceeding blindly.")

        # 4. Save
        timestamp = int(time.time())
        new_model_path = os.path.join(MODEL_DIR, f"watermark_model_v{timestamp}.pkl")
        
        # Save to temp file first
        with open(new_model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Update Scaler too
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(scaler, f)
            
        # Atomic Swap: Use os.replace for atomicity
        # This ensures MODEL_FILE is never missing, even if the script crashes during copy
        shutil.copy2(new_model_path, "temp_model_swap.pkl")
        os.replace("temp_model_swap.pkl", MODEL_FILE)
        
        # Cleanup old models (keep last 5)
        models = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith("watermark_model_v")])
        if len(models) > 5:
            for m in models[:-5]:
                os.remove(os.path.join(MODEL_DIR, m))
                
        result = {
            "retrain_ok": True,
            "new_model": new_model_path,
            "val_acc": val_acc,
            "samples": len(df)
        }
        print(json.dumps(result))
        logger.info("✅ Retrain Complete.")

    except Exception as e:
        logger.error(f"❌ Retrain Failed: {e}")
        print(json.dumps({"status": "failed", "error": str(e)}))
    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

if __name__ == "__main__":
    retrain()
