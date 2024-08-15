import base64
import wave
from typing import Final

from flask import Flask, jsonify, request
from google.cloud import language_v2, speech_v2

app = Flask(__name__)


MAX_AUDIO_MB: Final[str] = 5
default_config = speech_v2.RecognitionConfig(
    auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
    language_codes=["en-US"],
    model="long",
)
speech_client = speech_v2.SpeechClient()
language_client = language_v2.LanguageServiceClient()


@app.route("/sentiment/text", methods=["POST"])
def get_sentiment():
    if not request.is_json:
        return (
            jsonify(
                {"error": 'Invalid input. JSON body with "text" field is required.'}
            ),
            400,
        )
    data = request.json
    if "text" in data:
        text = data.get("text")
        sentiment = _analyze_sentiment(text=text)
        return jsonify(
            {
                "transcription": text,
                "sentiment": {
                    "score": sentiment.score,
                    "magnitude": sentiment.magnitude,
                },
            }
        )
    return jsonify({"error": 'Invalid JSON body, must contain "text" field.'}), 400


@app.route("/sentiment/audio", methods=["POST"])
def audio_sentiment():
    if "file" in request.files:
        audio_file = request.files.get("file")
        if not audio_file.filename.endswith(".wav"):
            return (
                jsonify({"error": "Invalid file type. Only .wav files supported."}),
                400,
            )
        with wave.open(audio_file, "rb") as wav_file:
            n_frames = wav_file.getnframes()
            audio_bytes = wav_file.readframes(n_frames)
    elif request.is_json:
        data = request.json
        if "audio" in data:
            try:
                audio_bytes = base64.b64decode(data.get("audio"))
            except Exception:
                return jsonify({"error": "Failed to decode audio data"}), 400
        else:
            return (
                jsonify({"error": 'Invalid JSON body, must contain an "audio" field.'}),
                400,
            )
    else:
        return (
            jsonify(
                {
                    "error": 'Invalid inpiut. Provide a .wav file or JSON body with field "audio".'
                }
            ),
            400,
        )
    try:
        text = _transcribe_audio(audio_bytes=audio_bytes)
        sentiment = _analyze_sentiment(text=text)
        return jsonify(
            {
                "transcription": text,
                "sentiment": {
                    "score": sentiment.score,
                    "magnitude": sentiment.magnitude,
                },
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


def _analyze_sentiment(text: str) -> language_v2.Sentiment:
    document = {"type_": language_v2.Document.Type.PLAIN_TEXT, "content": text}
    response = language_client.analyze_sentiment(document=document)
    return response.document_sentiment


def _transcribe_audio(
    audio_bytes: bytes,
    config: speech_v2.RecognitionConfig = default_config,
    limit_mb: int = MAX_AUDIO_MB,
) -> str:
    """Transcribe audio bytes to text using Google Speech-to-Text API"""
    limit = limit_mb << 20
    app.logger.info(limit)
    if len(audio_bytes) > limit:
        raise ValueError(
            f"The size of the speech data exceeds the limit of {limit_mb}MB"
        )
    request = speech_v2.RecognizeRequest(config=config, content=audio_bytes)
    response = speech_client.recognize(request=request)
    return response.results[0].alternatives[0].transcript if response.results else ""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
