# Use a slim Python base image
FROM python:3.10-slim-bullseye



# Set work directory inside container
WORKDIR /app

# Install system dependencies (FAISS needs BLAS + build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY . .

# Expose Flask port
EXPOSE 5000

# Default command to run app
# Make sure app.py runs Flask properly (e.g., with app.run(host="0.0.0.0", port=5000))
CMD ["python", "app.py"]
