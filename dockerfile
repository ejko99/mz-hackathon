FROM pytorch/pytorch
RUN mkdir /model
WORKDIR /model
COPY . .
RUN apt-get update
RUN apt-get -y install gcc
RUN pip install -r requirements.txt
