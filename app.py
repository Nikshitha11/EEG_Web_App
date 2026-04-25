from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from scipy.signal import resample   # ⭐ IMPORTANT (install scipy if missing)

app = Flask(__name__)

# -------- Load trained model and scaler --------
model = load_model("eeg_1dcnn_focal_nonfocal.h5")
scaler = joblib.load("eeg_scaler.pkl")

# -------- Home Page --------
@app.route("/")
def index():
    return render_template("index.html")

# -------- Prediction API (GENERALIZED INPUT) --------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # --- Get signal from frontend ---
        if not request.json or "signal" not in request.json:
            return jsonify({"error": "No EEG signal received"}), 400

        data = request.json["signal"]

        # Convert to numpy safely
        eeg = np.array(data, dtype=float)

        # Check empty signal
        if eeg.size == 0:
            return jsonify({"error": "Empty EEG signal"}), 400

        # ⭐ GENERALIZATION STEP
        # Convert ANY length signal → 10240 samples
        TARGET_LENGTH = 10240

        if eeg.shape[0] != TARGET_LENGTH:
            eeg = resample(eeg, TARGET_LENGTH)

        # --- Preprocess for model ---
        eeg = scaler.transform(eeg.reshape(1, -1))
        eeg = eeg.reshape(1, TARGET_LENGTH, 1)

        # --- Model prediction ---
        prob = float(model.predict(eeg, verbose=0)[0][0])
        label = "Focal EEG" if prob >= 0.5 else "Non-Focal EEG"

        return jsonify({"prediction": label})

    except Exception as e:
        print("ERROR:", e)   # shows error in terminal
        return jsonify({"error": str(e)}), 500

# -------- Run Flask --------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)