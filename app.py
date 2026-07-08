from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load trained models
stress_model = joblib.load("model/stress_model.pkl")
health_model = joblib.load("model/mental_health_model.pkl")


# Feature Engineering Function
def create_features(df):

    df = df.copy()

    # Social Media Intensity
    df["Social_Media_Intensity"] = (
        df["Avg_Daily_Usage_Hours"] *
        np.log1p(df["Daily_Unlocks"])
    )

    # Sleep Impact
    df["Sleep_Impact"] = (
        8 - df["Sleep_Hours_Per_Night"]
    )

    # Social Media Dependency
    df["Social_Media_Dependency"] = (
        df["Avg_Daily_Usage_Hours"] /
        (df["Study_Hours"] + 1)
    )

    # Healthy Lifestyle Score
    df["Healthy_Lifestyle_Score"] = (
        df["Sleep_Hours_Per_Night"] * 0.6 +
        df["Physical_Activity_Hours"] * 0.4
    )

    # Wellness Index
    df["Wellness_Index"] = (
        df["Sleep_Hours_Per_Night"]
        +
        df["Physical_Activity_Hours"]
        +
        df["Study_Hours"]
        -
        df["Avg_Daily_Usage_Hours"]
    )

    return df



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    # Collect data from HTML form

    data = {

        "Age": int(request.form["Age"]),

        "Gender": request.form["Gender"],

        "Country": request.form["Country"],

        "Academic_Level": request.form["Academic_Level"],

        "Most_Used_Platform": request.form["Most_Used_Platform"],

        "Purpose_Of_Use": request.form["Purpose_Of_Use"],


        "Avg_Daily_Usage_Hours":
        float(request.form["Avg_Daily_Usage_Hours"]),

        "Daily_Unlocks":
        int(request.form["Daily_Unlocks"]),


        "Study_Hours":
        float(request.form["Study_Hours"]),


        "Physical_Activity_Hours":
        float(request.form["Physical_Activity_Hours"]),


        "Sleep_Hours_Per_Night":
        float(request.form["Sleep_Hours_Per_Night"])

    }


    # Convert to DataFrame

    input_data = pd.DataFrame([data])


    # Add engineered features

    input_data = create_features(input_data)


    # Predictions

    stress_prediction = stress_model.predict(input_data)[0]

    health_prediction = health_model.predict(input_data)[0]


    # Round health score

    health_prediction = round(health_prediction,2)
    # Wellness Status
    if health_prediction >= 8:
        status = "Excellent Mental Wellness 🌟"
    elif health_prediction >= 6:
        status = "Good Mental Wellness 😊"
    elif health_prediction >= 4:
        status = "Moderate Mental Wellness 😐"
    else:
        status = "Needs Attention ⚠️"

# Stress Badge Color
    stress_colors = {
        "Low": "#22c55e",
        "Medium": "#eab308",
        "High": "#f97316",
        "Very High": "#ef4444"
    }

    stress_color = stress_colors.get(stress_prediction, "#6366f1")

    return render_template(
        "result.html",
        stress=stress_prediction,
        score=round(health_prediction,1),
        status=status,
        stress_color=stress_color
    )

    print("Stress:", stress_prediction)
    print("Score:", health_prediction)

    return render_template(
        "result.html",
        stress=stress_prediction,
        score=health_prediction,
        status=status,
        stress_color=stress_color
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)