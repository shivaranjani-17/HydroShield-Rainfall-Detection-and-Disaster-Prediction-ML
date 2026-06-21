# 🌧️ Rainfall Detection and Disaster Prediction Using Machine Learning

A web-based Machine Learning application that predicts rainfall levels and assesses disaster risks using meteorological parameters such as temperature, humidity, pressure, wind speed, and location.

The system uses Random Forest algorithms for rainfall prediction and disaster risk classification, providing early warnings and helping users make informed decisions for disaster preparedness.

---

## 📌 Features

- Predict rainfall amount in millimeters.
- Classify disaster risk levels.
- Real-time prediction through Flask web application.
- User-friendly interface for entering weather parameters.
- Displays nearby vulnerable areas during disaster risk situations.
- Fast predictions using pre-trained machine learning models.

---

## 🚀 Modules

### User Input Module
- Accepts location and weather parameters from users.

### Data Preprocessing Module
- Cleans and encodes input data.

### Rainfall Prediction Module
- Predicts rainfall amount using Random Forest Regressor.

### Disaster Risk Assessment Module
- Predicts disaster risk using Random Forest Classifier.

### Alert Generation Module
- Identifies and displays nearby vulnerable areas.

### Result Display Module
- Displays rainfall prediction and disaster risk results.

---

## 🛠️ Technologies Used

### Front-End
- HTML
- CSS
- Bootstrap

### Back-End
- Python
- Flask

### Machine Learning
- Scikit-learn
- Random Forest Regressor
- Random Forest Classifier

### Libraries
- Pandas
- NumPy
- Joblib

---

## 📂 Project Structure

```text
Code/
│
├── train_model.py
├── Train.ipynb
├── Prediction.ipynb
├── rainfall_disaster_prediction_dataset.xlsx
│
├── rainfall_predictor.pkl
├── disaster_risk_classifier.pkl
├── location_encoder.pkl
├── risk_encoder.pkl
│
└── app/
    │
    ├── app.py
    ├── static/
    ├── templates/
    │   ├── predict.html
    │   └── result.html
    │
    └── .ipynb_checkpoints/
```

---

## 📊 Dataset Features

The dataset contains approximately 2000 records with the following attributes:

- Location
- Temperature (°C)
- Humidity (%)
- Pressure (hPa)
- Wind Speed (km/h)
- Rainfall (mm)
- Disaster Risk

---

## 🤖 Machine Learning Models

### Rainfall Prediction
- Algorithm: Random Forest Regressor
- Output: Rainfall in millimeters

### Disaster Risk Prediction
- Algorithm: Random Forest Classifier
- Output Categories:
  - No Immediate Risk
  - Flood Risk Possible
  - Severe Flood Risk

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Rainfall-Detection-and-Disaster-Prediction-ML.git
```

### Install Required Libraries

```bash
pip install flask pandas numpy scikit-learn joblib
```

### Run the Flask Application

```bash
python app.py
```

### Open in Browser

```text
http://127.0.0.1:5000/
```

---

## 🎯 Objectives

- Improve rainfall forecasting accuracy.
- Provide early disaster warnings.
- Support disaster preparedness and management.
- Enable proactive decision-making.

---

## 🔮 Future Enhancements

- IoT sensor integration.
- Real-time weather API integration.
- SMS and Email alert system.
- Mobile application support.
- GIS-based visualization.
- Deep Learning-based forecasting.

---

## 👩‍💻 Author

**Shivaranjani A S**  
M.Sc. Software Systems

---

## 📜 License

This project is developed for academic and educational purposes only.
