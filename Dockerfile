FROM tensorflow/tensorflow:2.1.0-py3

RUN pip install dvc
RUN pip install 'dvc[gs]'
RUN mkdir -p /app
ADD train.py /app/