# Docker Guide for Forex AI

## Introduction to Docker

Docker is a platform for developing, shipping, and running applications in containers. Containers are lightweight, isolated environments that package an application and its dependencies, ensuring consistent behavior across different environments.

## Docker Components Used in Forex AI

The Forex AI Trading System uses Docker for:

1. **Containerization** - Running the application and its dependencies in isolated containers
2. **Docker Compose** - Orchestrating multiple containers as a single application
3. **Volume Management** - Persisting data across container restarts

## Installation

### Windows

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Enable WSL 2 (Windows Subsystem for Linux) if prompted
3. Start Docker Desktop

### macOS

1. Install [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
2. Start Docker Desktop

### Linux

```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

# Add Docker repository
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

# Update package index
sudo apt-get update

# Install Docker
sudo apt-get install docker-ce docker-ce-cli containerd.io

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.16.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

## Basic Docker Commands

```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# List running containers
docker ps

# List all containers (including stopped ones)
docker ps -a

# List images
docker images

# Pull an image
docker pull image_name:tag

# Run a container
docker run image_name:tag

# Stop a container
docker stop container_id

# Remove a container
docker rm container_id

# Remove an image
docker rmi image_name:tag
```

## Using Docker with Forex AI

### Starting the Application

```bash
# Navigate to project directory
cd forex_ai

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f
```

### Stopping the Application

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (caution: this will delete persistent data)
docker-compose down -v
```

### Rebuilding After Code Changes

```bash
# Rebuild containers
docker-compose build

# Rebuild and restart
docker-compose up -d --build
```

### Accessing Service Shells

```bash
# Access web service shell
docker-compose exec web bash

# Access PostgreSQL
docker-compose exec postgres psql -U forex_user -d forex_db

# Access Redis CLI
docker-compose exec redis redis-cli -a ${REDIS_PASSWORD}
```

## Docker Compose Configuration

The `docker-compose.yml` file defines the services used by Forex AI:

- **postgres**: PostgreSQL database with pgvector extension
- **redis**: Redis cache for real-time data
- **web**: The main web application
- **n8n**: Workflow automation engine

## Data Persistence

Docker volumes are used to persist data:

- **postgres_data**: PostgreSQL data
- **redis_data**: Redis data
- **n8n_data**: N8N workflows and credentials

## Troubleshooting

### Common Issues

1. **Port conflicts**
   - Change the port mapping in `docker-compose.yml`
   - Example: `"8001:8000"` instead of `"8000:8000"`

2. **Permission issues**
   - Ensure proper file permissions for mounted volumes
   - Run Docker commands with sudo on Linux if needed

3. **Container not starting**
   - Check logs: `docker-compose logs service_name`
   - Verify environment variables in `.env`

4. **Database connection issues**
   - Ensure PostgreSQL is healthy: `docker-compose ps`
   - Check network configuration

### Useful Debugging Commands

```bash
# View service logs
docker-compose logs -f service_name

# Inspect a container
docker inspect container_id

# Check network
docker network ls

# Inspect network
docker network inspect network_name

# View resource usage
docker stats
```

## Advanced Topics

### Custom Dockerfile

The project includes a custom Dockerfile for the web service, which:

1. Uses Python 3.11 as the base image
2. Installs system dependencies
3. Installs Python dependencies from `requirements.txt`
4. Installs TA-Lib (Technical Analysis Library)
5. Creates a non-root user for security

### Docker Networks

The Forex AI system uses a dedicated Docker network (`forex_network`) for communication between containers, providing:

- Isolated networking environment
- Automatic service discovery
- Secure communication

### Health Checks

Health checks ensure services are ready before dependent services start:

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U forex_user -d forex_db"]
  interval: 10s
  timeout: 5s
  retries: 5
```

## Further Learning

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [Docker Hub](https://hub.docker.com/) - Official Docker image repository 