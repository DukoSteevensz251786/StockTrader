"""
XGBoost pipeline on top of CNN embeddings
------------------------------------------
1. Loads the best CNN checkpoint
2. Extracts 128-dim embeddings for train/val/test
3. Trains XGBoost on those embeddings
4. Evaluates and saves the final model

Sentiment slot is clearly marked — when GDELT is ready,
add the sentiment column and retrain this script only.

Usage:
    python src/models/pipeline.py
"""

import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.training.dataset import get_splits
from src.models.cnn import CNN

# ── Config ─────────────────────────────────────────────────────────────────────
CNN_CHECKPOINT = Path("models/checkpoints/cnn_best.pt")
XGB_OUTPUT     = Path("models/checkpoints/xgb_pipeline.json")
BATCH_SIZE     = 512

device = (
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
print(f"Using device: {device}")

# ── Load CNN ───────────────────────────────────────────────────────────────────
print("\nLoading CNN checkpoint...")
model = CNN().to(device)
ckpt  = torch.load(CNN_CHECKPOINT, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"  Loaded epoch {ckpt['epoch']} — val acc {ckpt['val_acc']:.4f}")

# ── Extract embeddings ─────────────────────────────────────────────────────────
def extract_embeddings(dataset):
    """
    Run dataset through CNN and return:
        embeddings : np.ndarray (N, 128)
        labels     : np.ndarray (N,)
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_emb, all_lbl = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            emb = model.extract_features(x)   # (batch, 128)
            all_emb.append(emb.cpu().numpy())
            all_lbl.append(y.numpy())

    return np.concatenate(all_emb), np.concatenate(all_lbl)


print("\nExtracting CNN embeddings...")
train_ds, val_ds, test_ds = get_splits()

print("  Train  ", end="", flush=True)
X_train, y_train = extract_embeddings(train_ds)
print(f"→ {X_train.shape}")

print("  Val    ", end="", flush=True)
X_val, y_val = extract_embeddings(val_ds)
print(f"→ {X_val.shape}")

print("  Test   ", end="", flush=True)
X_test, y_test = extract_embeddings(test_ds)
print(f"→ {X_test.shape}")
# ── ADD SENTIMENT HERE ─────────────────────────────────────────────────────────
# sentiment_train = train_ds.get_sentiment_scores()
# sentiment_val   = val_ds.get_sentiment_scores()
# sentiment_test  = test_ds.get_sentiment_scores()
# X_train = np.hstack([X_train, sentiment_train.reshape(-1, 1)])
# X_val   = np.hstack([X_val,   sentiment_val.reshape(-1, 1)])
# X_test  = np.hstack([X_test,  sentiment_test.reshape(-1, 1)])
# ──────────────────────────────────────────────────────────────────────────────

# ── Train XGBoost ──────────────────────────────────────────────────────────────
print("\nTraining XGBoost...")
# ratio of down/up samples to balance the classes
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_model = xgb.XGBClassifier(
    n_estimators          = 500,
    max_depth             = 4,
    learning_rate         = 0.05,
    subsample             = 0.8,
    colsample_bytree      = 0.8,
    min_child_weight      = 10,
    gamma                 = 1.0,
    reg_lambda            = 2.0,
    scale_pos_weight      = scale_pos_weight,
    eval_metric           = "logloss",
    early_stopping_rounds = 20,
    device                = "cuda" if torch.cuda.is_available() else "cpu",
    random_state          = 42,
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

# ── Evaluate ───────────────────────────────────────────────────────────────────
def evaluate(X, y, name):
    preds = xgb_model.predict(X)
    acc   = accuracy_score(y, preds)
    print(f"\n{name} accuracy: {acc:.4f}")
    print(classification_report(y, preds, target_names=["DOWN", "UP"]))
    return acc

print("\n" + "="*50)
val_acc  = evaluate(X_val,  y_val,  "Validation")
test_acc = evaluate(X_test, y_test, "Test")

# ── Save ───────────────────────────────────────────────────────────────────────
XGB_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
xgb_model.save_model(XGB_OUTPUT)
print(f"\nXGBoost model saved to {XGB_OUTPUT.resolve()}")

# Save results summary
summary = {
    "cnn_val_acc"  : float(ckpt["val_acc"]),
    "xgb_val_acc"  : float(val_acc),
    "xgb_test_acc" : float(test_acc),
    "n_train"      : int(len(y_train)),
    "n_val"        : int(len(y_val)),
    "n_test"       : int(len(y_test)),
    "sentiment"    : False,   # flip to True after adding sentiment
}
summary_path = XGB_OUTPUT.parent / "results_summary.json"
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Results saved to    {summary_path.resolve()}")