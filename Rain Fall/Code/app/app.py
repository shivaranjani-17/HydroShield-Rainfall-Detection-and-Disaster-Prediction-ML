from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load models and encoders
reg = joblib.load("rainfall_predictor.pkl")
clf = joblib.load("disaster_risk_classifier.pkl")
le_location = joblib.load("location_encoder.pkl")
le_risk = joblib.load("risk_encoder.pkl")

# Nearby area map
nearby_map = {
    'Chennai': ['Tambaram', 'Chengalpattu', 'Kanchipuram'],
    'Madurai': ['Thiruparankundram', 'Melur', 'Thirumangalam'],
    'Coimbatore': ['Pollachi', 'Mettupalayam', 'Tirupur'],
    'Tiruchirappalli': ['Srirangam', 'Manapparai', 'Lalgudi'],
    'Salem': ['Mettur', 'Attur', 'Edappadi'],
    'Erode': ['Perundurai', 'Bhavani', 'Gobichettipalayam'],
    'Tirunelveli': ['Palayamkottai', 'Sankarankovil', 'Tenkasi'],
    'Vellore': ['Gudiyatham', 'Katpadi', 'Arakkonam'],
    'Thoothukudi': ['Kayalpattinam', 'Srivaikuntam', 'Eral'],
    'Dindigul': ['Oddanchatram', 'Palani', 'Nilakottai']
}

@app.route("/")
def home():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        location = request.form["location"]
        temperature = float(request.form["temperature"])
        humidity = int(request.form["humidity"])
        pressure = int(request.form["pressure"])
        wind_speed = float(request.form["wind_speed"])

        # Prepare input dataframe
        sample = pd.DataFrame([{
            'Location': location,
            'Temperature (°C)': temperature,
            'Humidity (%)': humidity,
            'Pressure (hPa)': pressure,
            'Wind Speed (km/h)': wind_speed
        }])

        # Encode location
        sample["Location"] = le_location.transform(sample["Location"])

        # Predict rainfall and risk
        rainfall_pred = reg.predict(sample)[0]
        risk_pred = clf.predict(sample)[0]
        disaster_label = le_risk.inverse_transform([risk_pred])[0]
        rainfall_rounded = round(rainfall_pred)

        # Nearby areas
        if disaster_label != "No Immediate Risk":
            nearby_areas = ', '.join(nearby_map.get(location, [])[:2])
        else:
            nearby_areas = "None"

        return render_template("result.html", rainfall=rainfall_rounded,
                               risk=disaster_label, nearby=nearby_areas)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
