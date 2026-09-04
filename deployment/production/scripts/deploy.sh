#!/bin/bash
# ==========================================
# 金融智能客服 Production Deployment Script
# ==========================================
# This script automates the deployment of the RAG chatbot to production
# Usage: ./deploy.sh [environment]
#   environment: dev (default) or prod

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV=${1:-dev}
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  金融智能客服 Deployment Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to print colored output
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    # Check Docker
    if ! command_exists docker; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"

    # Check Docker Compose
    if ! command_exists docker-compose; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"

    # Check if .env file exists
    if [ ! -f "$ENV_FILE" ]; then
        print_warning ".env file not found. Creating from .env.example..."
        if [ -f "$PROJECT_ROOT/.env.example" ]; then
            cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
            print_warning "Please edit .env file with your configuration before continuing"
            print_warning "Run this script again after configuring .env"
            exit 1
        else
            print_error ".env.example file not found. Please create .env file manually."
            exit 1
        fi
    fi
    print_success ".env file found"

    echo ""
}

# Function to backup current deployment
backup_deployment() {
    print_info "Creating backup of current deployment..."

    BACKUP_DIR="$PROJECT_ROOT/backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$BACKUP_DIR"

    # Backup database
    if docker-compose -f "$COMPOSE_FILE" ps db | grep -q "Up"; then
        print_info "Backing up database..."
        docker-compose -f "$COMPOSE_FILE" exec -T db pg_dump -U postgres finance_agent > "$BACKUP_DIR/db_backup.sql" 2>/dev/null || print_warning "Database backup failed"
        print_success "Database backed up to $BACKUP_DIR/db_backup.sql"
    fi

    # Backup .env file
    cp "$ENV_FILE" "$BACKUP_DIR/.env"
    print_success ".env file backed up"

    echo ""
}

# Function to build and start services
start_services() {
    print_info "Building and starting services..."

    # Use production compose file if prod environment
    if [ "$ENV" = "prod" ]; then
        COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"
        print_info "Using production configuration"
    fi

    # Pull latest images
    print_info "Pulling latest Docker images..."
    docker-compose -f "$COMPOSE_FILE" pull

    # Build images
    print_info "Building Docker images..."
    docker-compose -f "$COMPOSE_FILE" build --no-cache

    # Start services
    print_info "Starting services..."
    docker-compose -f "$COMPOSE_FILE" up -d

    print_success "Services started"
    echo ""
}

# Function to wait for services to be healthy
wait_for_services() {
    print_info "Waiting for services to be healthy..."

    # Wait for PostgreSQL
    print_info "Waiting for PostgreSQL..."
    for i in {1..30}; do
        if docker-compose -f "$COMPOSE_FILE" exec -T db pg_isready -U postgres >/dev/null 2>&1; then
            print_success "PostgreSQL is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "PostgreSQL did not become ready in time"
            exit 1
        fi
        sleep 2
    done

    # Wait for Redis
    print_info "Waiting for Redis..."
    for i in {1..30}; do
        if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping >/dev/null 2>&1; then
            print_success "Redis is ready"
            break
        fi
        if [ $i -eq 30 ]; then
            print_error "Redis did not become ready in time"
            exit 1
        fi
        sleep 2
    done

    # Wait for API
    print_info "Waiting for API service..."
    for i in {1..60}; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            print_success "API service is ready"
            break
        fi
        if [ $i -eq 60 ]; then
            print_error "API service did not become ready in time"
            print_warning "Check logs with: docker-compose -f $COMPOSE_FILE logs app"
            exit 1
        fi
        sleep 2
    done

    echo ""
}

# Function to run health checks
run_health_checks() {
    print_info "Running health checks..."

    # Check API health
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        print_success "API health check passed"
    else
        print_error "API health check failed"
        exit 1
    fi

    # Check database connection
    if docker-compose -f "$COMPOSE_FILE" exec -T db pg_isready -U postgres >/dev/null 2>&1; then
        print_success "Database health check passed"
    else
        print_error "Database health check failed"
        exit 1
    fi

    # Check Redis connection
    if docker-compose -f "$COMPOSE_FILE" exec -T redis redis-cli ping >/dev/null 2>&1; then
        print_success "Redis health check passed"
    else
        print_error "Redis health check failed"
        exit 1
    fi

    echo ""
}

# Function to print deployment info
print_deployment_info() {
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Deployment Successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Services:"
    echo "  - API: http://localhost:8000"
    echo "  - API Docs: http://localhost:8000/docs"
    echo "  - Database: localhost:5432"
    echo "  - Redis: localhost:6379"
    echo "  - Qdrant: http://localhost:6333"
    echo ""
    echo "Useful commands:"
    echo "  - View logs: docker-compose -f $COMPOSE_FILE logs -f"
    echo "  - Stop services: docker-compose -f $COMPOSE_FILE down"
    echo "  - Restart services: docker-compose -f $COMPOSE_FILE restart"
    echo ""
}

# Main deployment flow
main() {
    check_prerequisites

    if [ "$ENV" = "prod" ]; then
        backup_deployment
    fi

    start_services
    wait_for_services
    run_health_checks
    print_deployment_info
}

# Run main function
main
