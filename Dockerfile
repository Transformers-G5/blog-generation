# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /blog-generation

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

RUN python -m spacy download en_core_web_sm

COPY . .

CMD ["python", "src/app.py"]
#CMD ["gunicorn", "--workers=2", "--bind", "0.0.0.0:4040", "app:app"]