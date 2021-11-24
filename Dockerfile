FROM tensorflow/tensorflow

RUN apt-get -y update
RUN apt-get -y install git
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install dvc
RUN pip install 'dvc[gs]'
RUN pip install typing

RUN mkdir -p /app
ADD train.py /app/