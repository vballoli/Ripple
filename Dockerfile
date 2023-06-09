FROM python:3.10

WORKDIR /ripple

COPY . .
RUN pip install .

WORKDIR /experiments
