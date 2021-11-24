FROM tensorflow/tensorflow

RUN sudo apt install git-all
RUN /usr/bin/python3 -m pip install --upgrade pip
RUN pip install dvc
RUN pip install 'dvc[gs]'
RUN pip install typing

RUN mkdir -p /app
ADD train.py /app/