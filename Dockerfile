
FROM python:3.11-bullseye

LABEL description="ccres_disdrometer_processing docker image"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libudunits2-dev gdb gfortran \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY . /ccres_disdrometer_processing

RUN pip install /ccres_disdrometer_processing[dev]
