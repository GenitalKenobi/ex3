FROM python:3.9

WORKDIR /app

RUN pip install scikit-learn pandas numpy joblib

COPY . .

RUN python train.py

CMD ["python", "inference.py"]
