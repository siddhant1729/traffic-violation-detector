from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os

# --- Synthetic training data ---
# Features: [dist_pixels, fps, pixels_per_meter]
# Labels: "Normal" or "Overspeed"
X = np.array([
    [10, 30, 8],
    [12, 30, 8],
    [15, 30, 8],
    [25, 30, 8],
    [30, 30, 8],
    [6, 30, 8],
    [8, 30, 8],
    [28, 30, 8],
    [35, 30, 8],
    [40, 30, 8],
])
y = np.array(["Normal", "Normal", "Normal", "Overspeed", "Overspeed",
              "Normal", "Normal", "Overspeed", "Overspeed", "Overspeed"])

# --- Train Model ---
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X, y)

# --- Save Model ---
os.makedirs("ml_models", exist_ok=True)
joblib.dump(rf, "ml_models/overspeed_rf.pkl")

print("✅ Model trained and saved successfully at ml_models/overspeed_rf.pkl")
