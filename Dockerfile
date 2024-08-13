FROM python:3.9-slim

WORKDIR /ml-ops

COPY requirements.txt requirements.txt

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip install --upgrade pip \
    && pip install --no-cache-dir --ignore-installed -r requirements.txt \
    && apt-get purge -y python3-pip \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache


COPY . .
EXPOSE 8088

ENV PYTHONPATH "${PYTHONPATH}:/ml-ops/"

ENTRYPOINT ["/venv/bin/python3", "app/entry.py"]