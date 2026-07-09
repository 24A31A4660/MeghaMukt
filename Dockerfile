FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy python requirements and install them globally as root
COPY cloud-reconstruction/requirements.txt ./cloud-reconstruction/requirements.txt
# If backend has its own requirements, we should install them too
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r cloud-reconstruction/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories for runtime
RUN mkdir -p backend/uploads backend/outputs backend/data

# Set up a new user named "user" with user ID 1000 for Hugging Face
RUN useradd -m -u 1000 user

# Give the user ownership of the app directory so they can write to uploads/outputs
RUN chown -R user:user /app

# Switch to the "user" user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:

# Expose Hugging Face's required port
EXPOSE 7860

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]