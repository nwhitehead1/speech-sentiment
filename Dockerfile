FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8080

# Only needed for running locally. Do not push credentials JSON to compute!
# COPY credentials.json .
# ENV GOOGLE_APPLICATION_CREDENTIALS=
# ENV GCP_PROJECT_ID=
# ENV GCP_LOCATION=
# ENV GCP_RECOGNIZER=

CMD [ "gunicorn", "--workers", "2", "--bind", "0.0.0.0:8080", "app:app" ]