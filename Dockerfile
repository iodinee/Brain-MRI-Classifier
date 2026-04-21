FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "import gdown; gdown.download('https://drive.google.com/uc?id=1bAAlMoY_Bjq3h_P_T-8HHLZOogkugua3', 'brain_tumor_model.h5', quiet=False)"

EXPOSE 8080
CMD ["python", "app.py"]