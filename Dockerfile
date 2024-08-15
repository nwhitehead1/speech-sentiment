FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

# Only needed for running locally. Do not push credentials JSON to compute!
# ENV GOOGLE_APPLICATION_CREDENTIALS="credentials.json"

CMD [ "python", "app.py" ]