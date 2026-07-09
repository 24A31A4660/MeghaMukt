FROM python:3.10-slim

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy python requirements and install them
COPY cloud-reconstruction/requirements.txt ./cloud-reconstruction/requirements.txt
RUN pip install --no-cache-dir -r cloud-reconstruction/requirements.txt

# Copy the rest of the application
COPY . .

# Install Node.js backend dependencies
WORKDIR /app/backend
RUN npm install

# Set working directory back to root
WORKDIR /app

# Create necessary directories for runtime
RUN mkdir -p backend/uploads backend/outputs

# Expose the backend port
EXPOSE 8000

# Start the Node.js server
CMD ["node", "backend/server.js"]
