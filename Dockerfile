# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# COPY ./requirements.txt /app/requirements.txt

# RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# COPY . .

# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]