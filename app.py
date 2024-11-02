from flask import Flask, request, render_template, jsonify
import numpy as np
import pickle
import warnings
from feature import FeatureExtraction  # Ensure feature.py is in the same directory or correct path

warnings.filterwarnings('ignore')

# Load the model
with open("pickle/model.pkl", "rb") as file:
    gbc = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")  # Get URL input from the form
        if url:
            # Extract features using FeatureExtraction
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, -1)  # Reshape as needed by the model

            # Make predictions
            y_pred = gbc.predict(x)[0]  # 1 is safe, -1 is unsafe
            y_pro_phishing = gbc.predict_proba(x)[0, 0]  # Probability of phishing
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]  # Probability of non-phishing

            # Interpret the prediction
            print(y_pred)
            print(y_pro_non_phishing)
            if y_pred == 1:
                pred = f"It is {y_pro_non_phishing * 100:.2f}% safe to go"
            else:
                pred = f"Warning! This site has a {y_pro_phishing * 100:.2f}% chance of being phishing."

            # Pass results to the template
            return render_template('index.html', prediction=pred, url=url, 
                                   pro_safe=round(y_pro_non_phishing * 100, 2), 
                                   pro_phishing=round(y_pro_phishing * 100, 2))

    # GET request, or if no URL is provided
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
