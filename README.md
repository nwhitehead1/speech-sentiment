# 1984-team1

### Testing the Backend locally

Here is the criteria for running the backend locally:

1. Make sure you have the following environment variables set:
 - `GOOGLE_APPLICATION_CREDENTIALS`: Path to google credentials JSON
 - `GCP_PROJECT_ID`: The project id for the google cloud account - be sure the proper APIs are enabled (Language, and Speech-to-Text)
 - `GCP_LOCATION`: The region
 - `GCP_RECOGNIZER`: A recognizer that is enabled and running
2. Create a `.wav` file
3. Make sure the file is using the correct encoding, by running:
```bash
ffmpeg -i /path/to/file.wav -f wav -ar 16000 file.wav
```
This ensures the proper encoding and sample rate. For some reason auto-detect isn't working with speech_v2!
4. Start the Flask server
5. Create a request
```bash
curl -X POST -F file=@/path/to/file.wav http://localhost:8080/sentiment/audio
```
