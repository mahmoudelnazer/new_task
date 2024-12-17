# Use Ubuntu instead of Debian slim
FROM ubuntu:22.04

# Set the working directory in the container
WORKDIR /app

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package sources and install dependencies
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update --fix-missing && \
    apt-get install -y \
    build-essential \
    curl \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Expose the ports for FastAPI and Streamlit
EXPOSE 8000 8501

# The command will be specified in docker-compose.yml