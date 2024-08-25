import base64
import logging
import os
import wave
from typing import Final

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from google.cloud import language_v2, logging_v2, speech_v2, texttospeech
from scipy.fft import fft

app = Flask(__name__)
CORS(app)

logging_client = logging_v2.Client()
logging_client.setup_logging(log_level=logging.INFO)

cloud_handler = logging_v2.handlers.CloudLoggingHandler(client=logging_client)
app.logger.addHandler(cloud_handler)

MAX_AUDIO_MB: Final[str] = 10
DEFAULT_ENCODING: Final[int] = (
    speech_v2.types.ExplicitDecodingConfig.AudioEncoding.LINEAR16
)
DEFAULT_SAMPLE_RATE: Final[int] = 25000
DEFAULT_CHANNEL_COUNT: Final[int] = 1
LOW_FREQUENCY_THRESHOLD: Final[int] = 250
HIGH_FREQUENCY_THRESHOLD: Final[int] = 4000
default_config = speech_v2.RecognitionConfig(
    # auto_decoding_config=speech_v2.AutoDetectDecodingConfig()
    explicit_decoding_config=speech_v2.ExplicitDecodingConfig(
        encoding=DEFAULT_ENCODING,
        sample_rate_hertz=DEFAULT_SAMPLE_RATE,
        audio_channel_count=DEFAULT_CHANNEL_COUNT,
    ),
    language_codes=["en-US"],
    model="long",
)
gcp_project_id = os.environ["GCP_PROJECT_ID"]
gcp_location = os.environ.get("GCP_LOCATION", default="global")
gcp_recognizer = os.environ["GCP_RECOGNIZER"]
default_recognizer = (
    f"projects/{gcp_project_id}/locations/{gcp_location}/recognizers/{gcp_recognizer}"
)
speech_client = speech_v2.SpeechClient()
language_client = language_v2.LanguageServiceClient()
text_to_speech_client = texttospeech.TextToSpeechClient()


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


@app.route("/sentiment/synthesize", methods=["POST"])
def get_synthesized_speech():
    if not request.is_json:
        return (
            jsonify(
                {"error": 'Invalid input. JSON body with "text" field is required.'}
            ),
            400,
        )
    data = request.json
    text = data.get("text")
    if not text:
        return (
            jsonify(
                {"error": 'Invalid input. JSON body with "text" field is required.'}
            ),
            400,
        )
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-GB",
        # https://cloud.google.com/text-to-speech/docs/voices
        name="en-GB-Standard-D",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16, speaking_rate=0.8, pitch=2.0
    )
    text_to_speech_request = texttospeech.SynthesizeSpeechRequest(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    response = text_to_speech_client.synthesize_speech(request=text_to_speech_request)
    encoded_content = base64.b64encode(response.audio_content).decode("utf-8")
    return (jsonify({"content": encoded_content}), 200)


@app.route("/sentiment/audio", methods=["POST"])
def audio_sentiment():
    if "file" in request.files:
        audio_file = request.files.get("file")
        app.logger.info(
            f"Loading file {audio_file.filename} with MIME type {audio_file.mimetype}"
        )
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
                    "error": 'Invalid input. Provide a .wav file or JSON body with field "audio".'
                }
            ),
            400,
        )
    try:
        tone: Tone = _extract_tone(audio_bytes=audio_bytes)
        transcription: Transcription = _transcribe_audio(audio_bytes=audio_bytes)
        app.logger.info(
            f"Transcription - {transcription.transcript} (confidence: {transcription.confidence})"
        )
        sentiment: language_v2.Sentiment = _analyze_sentiment(
            text=transcription.transcript
        )
        app.logger.info(
            f"Sentiment - score: {sentiment.score}, magnitude: {sentiment.magnitude}"
        )
        script = (
            transcription.transcript if transcription.transcript else "No transcription"
        )
        script_confidence = transcription.confidence if transcription.confidence else 0
        return jsonify(
            {
                "transcription": {
                    "transcript": script,
                    "confidence": script_confidence,
                },
                "tone": {
                    "peak_frequency": tone.frequency,
                    "classification": tone.classification(),
                },
                "sentiment": {
                    "score": sentiment.score,
                    "magnitude": sentiment.magnitude,
                },
            }
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


class Tone:
    def __init__(self, frequency: float):
        self._frequency = frequency

    @property
    def frequency(self) -> float:
        return self._frequency

    def classification(self) -> str:
        if self._frequency < LOW_FREQUENCY_THRESHOLD:
            return "LOW"
        elif LOW_FREQUENCY_THRESHOLD <= self._frequency < HIGH_FREQUENCY_THRESHOLD:
            return "MEDIUM"
        return "HIGH"


class Transcription:
    def __init__(self, transcript: str, confidence: float):
        app.logger.info(f"Transcription: {transcript} (confidence: {confidence})")
        self._transcript = transcript
        self._confidence = confidence

    @property
    def transcript(self) -> str:
        return self._transcript

    @property
    def confidence(self) -> float:
        return self._confidence


def _analyze_sentiment(text: str) -> language_v2.Sentiment:
    document = {"type_": language_v2.Document.Type.PLAIN_TEXT, "content": text}
    app.logger.info("analyzing:" + text)
    response = language_client.analyze_sentiment(document=document)
    app.logger.info(f"Sentiment analysis response: {response}")
    return response.document_sentiment


def _transcribe_audio(
    audio_bytes: bytes,
    config: speech_v2.RecognitionConfig = default_config,
    limit_mb: int = MAX_AUDIO_MB,
    recognizer: str = default_recognizer,
) -> Transcription:
    """Transcribe audio bytes to text using Google Speech-to-Text API"""
    limit = limit_mb << 20
    if len(audio_bytes) > limit:
        return jsonify(
            {"error": f"The size of the speech data exceeds the limit of {limit_mb}MB"}
        )
    request = speech_v2.RecognizeRequest(
        recognizer=recognizer, config=config, content=audio_bytes
    )
    response = speech_client.recognize(request=request)
    app.logger.info(f"Response: {response}")
    try:
        alternatives = response.results[0].alternatives[0]
        script = (
            alternatives.transcript if alternatives.transcript else "No transcription"
        )
        script_conf = alternatives.confidence if alternatives.confidence else 0
        app.logger.info(f"Transcription: {script} (confidence: {script_conf})")
        return Transcription(transcript=script, confidence=script_conf)
    except (IndexError, AttributeError) as e:
        return jsonify({"error": f"Error accessing response data: {str(e)}"})


def _extract_tone(audio_bytes: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> Tone:
    audio_data = np.frombuffer(audio_bytes, dtype=np.int8)
    # Frequency analysis - looking for intonation
    audio_data = audio_data / np.max(np.abs(audio_data))
    n = len(audio_data)
    y_freq = fft(audio_data)
    x_freq = np.fft.fftfreq(n, 1 / sample_rate)
    peak_freq_idx = np.argmax(np.abs(y_freq))
    peak_freq = x_freq[peak_freq_idx]
    return Tone(frequency=peak_freq)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
