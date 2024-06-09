FROM tensorflow/tensorflow:2.16.1-gpu-jupyter


WORKDIR /usr/local/dev

COPY ./requirements.txt requirements.txt

RUN pip install -r requirements.txt 

EXPOSE 8888

ENTRYPOINT ["jupyter","lab","--ip=0.0.0.0","--allow-root","--no-browser"]