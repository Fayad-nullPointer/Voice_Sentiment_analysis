import whisper
from transformers import pipeline
from flask import Flask, request, jsonify
import threading
import os
import uuid
import subprocess
# Whisper model
whisper_model = whisper.load_model("base")

# HuggingFace sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    # Save uploaded file with random name
    audio = request.files["audio"]
    audio_id = str(uuid.uuid4())
    input_path = f"{audio_id}_input"
    output_path = f"{audio_id}.wav"
    audio.save(input_path)

    try:
        # Convert to .wav using ffmpeg
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # Transcribe using Whisper
        text = whisper_model.transcribe(output_path)["text"]

        # Sentiment analysis
        sentiment = sentiment_pipeline(text)[0]
        label = sentiment['label'].upper()
        score = sentiment['score']
        rating_map = {"POSITIVE": 5, "NEGATIVE": 1, "NEUTRAL": 2.5}
        rating = rating_map.get(label, 3)

        return jsonify({
            "transcription": text,
            "sentiment": {
                "label": label,
                "score": score
            },
            "rating": rating
        })

    except Exception as e:
        return jsonify({"error": str(e)})

    finally:
        # Clean up files
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

# Start Flask app in background thread
def run_flask():
    app.run(port=5000)

thread = threading.Thread(target=run_flask)
thread.start()
