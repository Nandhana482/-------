from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load saved models
log_model = joblib.load("classification_model.pkl")
rf_model = joblib.load("random_forest.pkl")
xgb_model = joblib.load("xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        float(request.form["age"]),
        float(request.form["gender"]),
        float(request.form["sleep_hours"]),
        float(request.form["sleep_quality"]),
        float(request.form["social_interaction_days"]),
        float(request.form["living_alone"]),
        float(request.form["work_hours_weekly"]),
        float(request.form["work_stress_level"]),
        float(request.form["screen_time_hours"]),
        float(request.form["family_support_score"]),
        float(request.form["therapy_access"]),
        float(request.form["mood_swing_frequency"]),
        float(request.form["appetite_changes"])
    ]

    data = np.array(data).reshape(1, -1)

    # Logistic Regression needs scaling
    scaled_data = scaler.transform(data)

    # Predictions
    pred_log = log_model.predict(scaled_data)[0]
    pred_rf = rf_model.predict(data)[0]
    pred_xgb = xgb_model.predict(data)[0]

    return render_template(
        "index.html",
        logistic=pred_log,
        randomforest=pred_rf,
        xgboost=pred_xgb
    )

if __name__ == "__main__":
    app.run(debug=True)
