FROM ubuntu:noble

ENV PYTHONUNBUFFERED=1

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get -yq install python3-pip python3-venv

WORKDIR /cw3

COPY *.py /cw3/

RUN mkdir -p /cw3/logs /cw3/db /cw3/model /cw3/data /data

COPY /model/aki_model.pkl /cw3/model/
COPY /db/schema.sql /cw3/db/

RUN python3 -m venv /cw3
COPY requirements.txt /cw3/
RUN /cw3/bin/pip install -r /cw3/requirements.txt

EXPOSE 8440
EXPOSE 8441
EXPOSE 8000

WORKDIR /cw3
CMD ["/cw3/bin/python", "-m", "main"]