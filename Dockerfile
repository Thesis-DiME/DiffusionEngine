FROM python:3.11-slim

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
 chmod u+x ./setup.sh && ./setup.sh

COPY . .

CMD [ "python", "main.py" ]

