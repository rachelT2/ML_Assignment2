FROM python:3.11.5-bookworm

RUN pip install --upgrade pip

WORKDIR /root/app

COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt

COPY code/ code/
WORKDIR /root/app/code
CMD [ "python3", "app.py" ]