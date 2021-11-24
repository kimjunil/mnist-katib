FROM tensorflow/tensorflow

RUN apt-get update && apt-get install -y git
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install dvc
RUN pip install 'dvc[gs]'
RUN pip install typing

RUN mkdir -p /app
ADD train.py /app/