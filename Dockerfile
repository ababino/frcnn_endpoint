FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime


WORKDIR /usr/src/app
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY apis/ apis/

EXPOSE 4020

CMD ["gunicorn", "--bind", "0.0.0.0:4020",  "app:app"]
