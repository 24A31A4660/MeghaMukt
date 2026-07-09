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

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Switch to the "user" user
USER user

# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:

# Set the working directory to the user's home directory
WORKDIR C:\Users\aksha/app

# Copy requirements and install
COPY --chown=user cloud-reconstruction/requirements.txt C:\Users\aksha/app/cloud-reconstruction/requirements.txt
RUN pip install --no-cache-dir -r cloud-reconstruction/requirements.txt

# Copy the rest of the app with correct permissions
COPY --chown=user . C:\Users\aksha/app

# Install Node.js backend dependencies
WORKDIR C:\Users\aksha/app/backend
RUN npm install

# Set working directory back to root
WORKDIR C:\Users\aksha/app

# Create necessary directories for runtime
RUN mkdir -p backend/uploads backend/outputs

# Start the Node.js server
CMD ["node", "backend/server.js"]