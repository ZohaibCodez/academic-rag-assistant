FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (better caching)
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt 

# Copy only necessary application files
COPY app.py ./

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]