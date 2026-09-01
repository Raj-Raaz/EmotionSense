from flask import Flask, render_template, request
import requests

app = Flask(__name__)

API_URL = "http://127.0.0.1:8000/predict"


@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    emotion = None
    confidence = None

    if request.method == "POST":
        text = request.form["text"]

        response = requests.post(API_URL, json={"text": text})
        result = response.json()

        emotion = result["emotion"]
        confidence = result["confidence"]

    return render_template(
        "index.html",
        text=text,
        emotion=emotion,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)