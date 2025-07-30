# Use the official PyTorch image with CUDA support as the base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set environment variables to prevent Python from writing .pyc files and to ensure stdout/stderr are unbuffered
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set DEBIAN_FRONTEND to noninteractive to suppress interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Preconfigure tzdata to set the time zone to UTC
RUN echo "tzdata tzdata/Areas select Etc" | debconf-set-selections && \
    echo "tzdata tzdata/Zones/Etc select UTC" | debconf-set-selections

# Install system dependencies, including Git, libGL, and tzdata
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libgl1-mesa-glx \      
    git \
    tzdata \                
    && rm -rf /var/lib/apt/lists/*

# Reset DEBIAN_FRONTEND to default
ENV DEBIAN_FRONTEND=dialog

# Create a non-root user with specific UID and GID
RUN groupadd -g 1001 appgroup && \
    useradd -m -u 1001 -g appgroup appuser

# Set the working directory
WORKDIR /app

# Create the 'uploads' directory and set ownership
RUN mkdir -p uploads && chown appuser:appgroup uploads

# Switch to the non-root user
USER appuser

# Copy and install Python dependencies
COPY --chown=appuser:appgroup requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=appuser:appgroup . /app/

# Expose the port your application runs on
EXPOSE 1412

# Define environment variable for Flask
ENV FLASK_APP=app.py

# Run the application using Gunicorn for better process management
CMD ["python", "app.py"]

