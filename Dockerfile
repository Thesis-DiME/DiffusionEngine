FROM python:3.11-slim

WORKDIR /app

COPY setup.sh requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN --mount=type=cache,target=/root/.cache/pip \
 chmod u+x ./setup.sh && ./setup.sh

COPY . .

CMD [ "python", "main.py" ]

