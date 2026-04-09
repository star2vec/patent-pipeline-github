FROM python:3.10-slim

# Prevent Python from writing .pyc files and force stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (like your ffmpeg/curl if you still need them)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code into the container
COPY . .

# Expose the port the Flask web server will run on
EXPOSE 5000

# Set the default command to run the Flask server
CMD ["python", "server.py"]