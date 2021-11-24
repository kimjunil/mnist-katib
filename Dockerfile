FROM tensorflow/tensorflow

RUN pip install dvc
RUN pip install 'dvc[gs]'
RUN pip install typing

RUN mkdir -p /app
ADD train.py /app/