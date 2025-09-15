from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import pickle
import smtplib

# ---- Load model and dataset ----
model = pickle.load(open("aqi_model.pkl", "rb"))
data = pd.read_csv("area_pollution_data.csv")

app = Flask(__name__)
app.secret_key = "your_secret_key"   # needed for flash messages

# Keep this consistent with how you trained the model
features_columns = ['PM2.5', 'PM10', 'NO2', 'CO', 'OZONE']

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 200:
        return "Poor", "orange"
    else:
        return "Hazardous", "red"

@app.route("/", methods=["GET", "POST"])
def index():
    # aqi_result, category, color = None, None, None

    # # unique station list for dropdown
    # areas = sorted(data['station'].dropna().unique().tolist())
    
    prediction = None
    if request.method == "POST":
        # selected_station = request.form.get("station")
        station = request.form.get("station")
        if station:
            # get one row for that station with all required pollutant values
            try:
                features = data.loc[data['station'] == station, features_columns].values
                if len(features)>0:
                    prediction = model.predict(features)[0]
                else:
                    prediction = "No data this station"
            except Exception as e:
                prediction = f"Error: {e}"

    return render_template(
        "index.html",
        areas=data['station'].unique(),
        prediction=prediction
    )


@app.route("/manual", methods=["GET", "POST"])
def manual():
    prediction, category, color = None, None, None
    if request.method == "POST":
        try:
            # keep order consistent with training
            pm25 = float(request.form["pm25"])
            pm10 = float(request.form["pm10"])
            no2  = float(request.form["no2"])
            so2  = float(request.form["so2"])
            co   = float(request.form["co"])
            o3   = float(request.form["o3"])

            features = [[pm25, pm10, no2, so2, co, o3]]
            prediction = round(float(model.predict(features)[0]), 2)
            category, color = get_aqi_category(prediction)
            
        except Exception as e:
            flash(f"Error: {e}", "danger")
    return render_template("manual.html", prediction=prediction, category=category, color=color)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        message = request.form["message"]
        try:
            sender_email = "yourgmail@gmail.com"
            sender_password = "your_app_password"
            receiver_email = "debashismoharana09@gmail.com"
            subject = f"Contact Form Message from {name}"
            body = f"From: {name}\nEmail: {email}\n\nMessage:\n{message}"
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, receiver_email, f"Subject: {subject}\n\n{body}")
            flash("Message sent successfully!", "success")
        except Exception as e:
            flash(f"Error: {e}", "danger")
        return redirect(url_for("contact"))
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True)
