#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if gcloud is installed
if ! command_exists gcloud; then
    echo "Error: gcloud CLI is not installed. Please install it first."
    exit 1
fi

# Check if user is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" >/dev/null 2>&1; then
    echo "Error: Not authenticated with gcloud. Please run 'gcloud auth login' first."
    exit 1
fi

# Set variables
PROJECT_ID=$(gcloud config get-value project)
SERVICE_NAME="forex-ai"
REGION="us-central1"  # Change this if needed

echo "Setting up environment variables for Cloud Run service: $SERVICE_NAME"
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"

# Create secrets if they don't exist
create_secret_if_not_exists() {
    local secret_name=$1
    local secret_value=$2
    
    if ! gcloud secrets describe "$secret_name" >/dev/null 2>&1; then
        echo "Creating secret: $secret_name"
        echo -n "$secret_value" | gcloud secrets create "$secret_name" --data-file=-
    else
        echo "Secret already exists: $secret_name"
    fi
}

# Function to prompt for sensitive values
get_sensitive_value() {
    local prompt=$1
    local value=""
    while [ -z "$value" ]; do
        echo -n "$prompt: "
        read -s value
        echo
    done
    echo "$value"
}

# Get sensitive values
SUPABASE_URL=$(get_sensitive_value "Enter Supabase URL")
SUPABASE_KEY=$(get_sensitive_value "Enter Supabase Key")
SUPABASE_SERVICE_KEY=$(get_sensitive_value "Enter Supabase Service Key")
JWT_SECRET_KEY=$(get_sensitive_value "Enter JWT Secret Key")
SECRET_KEY=$(get_sensitive_value "Enter Secret Key")

# Create secrets
create_secret_if_not_exists "supabase-url" "$SUPABASE_URL"
create_secret_if_not_exists "supabase-key" "$SUPABASE_KEY"
create_secret_if_not_exists "supabase-service-key" "$SUPABASE_SERVICE_KEY"
create_secret_if_not_exists "jwt-secret" "$JWT_SECRET_KEY"
create_secret_if_not_exists "secret-key" "$SECRET_KEY"

# Update Cloud Run service with environment variables and secrets
echo "Updating Cloud Run service with environment variables and secrets..."
gcloud run services update $SERVICE_NAME \
    --region=$REGION \
    --project=$PROJECT_ID \
    --set-env-vars="ENVIRONMENT=production,DEBUG=false,LOG_LEVEL=INFO,ENABLE_DOCS=true" \
    --set-secrets="SUPABASE_URL=supabase-url:latest,SUPABASE_KEY=supabase-key:latest,SUPABASE_SERVICE_KEY=supabase-service-key:latest,JWT_SECRET_KEY=jwt-secret:latest,SECRET_KEY=secret-key:latest"

echo "Environment setup complete!" 