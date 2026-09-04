#!/bin/bash

# Development helper script for 金融智能客服

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage
show_usage() {
    cat << EOF
Usage: ./scripts/dev.sh <command>

Commands:
    install     Install dependencies
    test        Run tests
    test-cov    Run tests with coverage
    lint        Run linters (ruff, mypy)
    format      Format code with ruff
    dev         Start development server
    migrate     Run database migrations
    makemig     Create new migration
    docker      Start Docker services
    docker-down Stop Docker services
    clean       Clean up temporary files

Examples:
    ./scripts/dev.sh install
    ./scripts/dev.sh test
    ./scripts/dev.sh dev
EOF
}

# Install dependencies
install_deps() {
    print_info "Installing dependencies..."
    pip install -e ".[dev]"
    print_info "Dependencies installed successfully!"
}

# Run tests
run_tests() {
    print_info "Running tests..."
    pytest
}

# Run tests with coverage
run_tests_coverage() {
    print_info "Running tests with coverage..."
    pytest --cov=app --cov-report=html --cov-report=term
    print_info "Coverage report generated in htmlcov/index.html"
}

# Run linters
run_lint() {
    print_info "Running ruff linter..."
    ruff check app/

    print_info "Running mypy type checker..."
    mypy app/
}

# Format code
format_code() {
    print_info "Formatting code with ruff..."
    ruff format app/
    print_info "Code formatted!"
}

# Start development server
start_dev() {
    print_info "Starting development server..."
    uvicorn app.main:app --reload --log-level debug
}

# Run migrations
run_migrations() {
    print_info "Running database migrations..."
    alembic upgrade head
    print_info "Migrations completed!"
}

# Create new migration
create_migration() {
    if [ -z "$1" ]; then
        print_error "Please provide a migration message"
        echo "Usage: ./scripts/dev.sh makemig '<message>'"
        exit 1
    fi
    print_info "Creating new migration: $1"
    alembic revision --autogenerate -m "$1"
}

# Start Docker services
start_docker() {
    print_info "Starting Docker services..."
    docker-compose -f deployment/docker/docker-compose.yml up -d
    print_info "Docker services started!"
    print_info "App: http://localhost:8000"
    print_info "API Docs: http://localhost:8000/docs"
}

# Stop Docker services
stop_docker() {
    print_info "Stopping Docker services..."
    docker-compose -f deployment/docker/docker-compose.yml down
    print_info "Docker services stopped!"
}

# Clean temporary files
clean_temp() {
    print_info "Cleaning temporary files..."
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete
    find . -type f -name "*.pyo" -delete
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
    rm -rf htmlcov/ .coverage
    print_info "Cleanup complete!"
}

# Main script logic
case "$1" in
    install)
        install_deps
        ;;
    test)
        run_tests
        ;;
    test-cov)
        run_tests_coverage
        ;;
    lint)
        run_lint
        ;;
    format)
        format_code
        ;;
    dev)
        start_dev
        ;;
    migrate)
        run_migrations
        ;;
    makemig)
        create_migration "$2"
        ;;
    docker)
        start_docker
        ;;
    docker-down)
        stop_docker
        ;;
    clean)
        clean_temp
        ;;
    *)
        show_usage
        exit 1
        ;;
esac
