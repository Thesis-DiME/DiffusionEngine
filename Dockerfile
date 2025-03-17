FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN --mount=type=cache,target=/root/.cache/pip \
 chmod u+x ./setup.sh && ./setup.sh

CMD [ "python", "main.py" ]

