import base64
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


@app.route("/sentiment", methods=["GET"])
def get_sentiment():
    text = request.args.get("text")
    if not text:
        return jsonify({"error": "No text data provided"}), 400
    sentiment = _analyze_sentiment(text=text)
    return jsonify(
        {
            "transcription": text,
            "sentiment": {"score": sentiment.score, "magnitude": sentiment.magnitude},
        }
    )


@app.route("/sentiment/audio", methods=["GET"])
def audio_sentiment():
    audio = request.args.get("audio")
    if not audio:
        return jsonify({"error": "No audio data provided"}), 400
    try:
        audio_bytes = base64.b64decode(audio)
    except Exception:
        return jsonify({"error": "Failed to decode audio data"}), 400
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
    lang_client = language_v2.LanguageServiceClient()
    document = {"type_": language_v2.Document.Type.PLAIN_TEXT, "content": text}
    response = lang_client.analyze_sentiment(document=document)
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
    client = speech_v2.SpeechClient()
    request = speech_v2.RecognizeRequest(config=config, content=audio_bytes)
    response = client.recognize(request=request)
    return response.results[0].alternatives[0].transcript if response.results else ""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
