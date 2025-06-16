# FROM python:3.10

# WORKDIR /app

# COPY requirements.txt ./
# RUN pip install --no-cache-dir -r requirements.txt

# COPY . .

# EXPOSE 5000

# # Run the MLflow UI
# CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]

# # Run training during image build
# RUN python /app/src/train.py

# # Start the Flask API for inference
# CMD ["python", "app/src/app.py"]

FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

python /app/src/train.py

# EXPOSE 5000 # MLflow
# EXPOSE 8000 # Flask

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
