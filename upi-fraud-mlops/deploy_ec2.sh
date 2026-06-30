#!/bin/bash
# EC2 Deployment Script for Ubuntu
set -e

echo "Starting EC2 Deployment Setup..."

# Update packages
sudo apt-get update -y

# Install Docker if not installed
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    sudo apt-get install -y ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi

# Ensure docker-compose is available (docker compose plugin is installed above)
echo "Starting the application stack..."
sudo docker compose down || true
sudo docker compose up -d --build

echo "Deployment successful!"
echo "Services are running:"
echo "- FastAPI: Port 8000"
echo "- Streamlit Dashboard: Port 8501"
echo "- MLflow: Port 5001"
