FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    default-libmysqlclient-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Pre-install pandas and numpy before sdv to prevent dependency resolver
# from pulling incompatible versions that require compilation
RUN pip install --no-cache-dir pandas==2.2.2 numpy==1.26.4
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 7860

CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn trialguard.wsgi:application --bind 0.0.0.0:7860 --workers 1 --timeout 120"]
