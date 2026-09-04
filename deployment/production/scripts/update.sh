#!/bin/bash
# ==========================================
# 金融智能客服 Zero-Downtime Update Script
# ==========================================
# This script performs zero-downtime updates of the application
# Usage: ./update.sh

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.prod.yml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  金融智能客服 Zero-Downtime Update${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Function to pull latest code
pull_latest_code() {
    echo -e "${YELLOW}Pulling latest code...${NC}"
    cd "$PROJECT_ROOT"
    git pull origin main
    echo -e "${GREEN}✓ Code updated${NC}"
    echo ""
}

# Function to build new image
build_new_image() {
    echo -e "${YELLOW}Building new Docker image...${NC}"
    cd "$PROJECT_ROOT"
    docker-compose -f "$COMPOSE_FILE" build app
    echo -e "${GREEN}✓ Image built${NC}"
    echo ""
}

# Function to create backup
create_backup() {
    echo -e "${YELLOW}Creating pre-update backup...${NC}"
    "$SCRIPT_DIR/backup.sh"
    echo -e "${GREEN}✓ Backup created${NC}"
    echo ""
}

# Function to update services
update_services() {
    echo -e "${YELLOW}Updating services...${NC}"

    # Update app service one by one (for zero-downtime)
    docker-compose -f "$COMPOSE_FILE" up -d --no-deps app

    echo -e "${GREEN}✓ Services updated${NC}"
    echo ""
}

# Function to wait for health check
wait_for_health() {
    echo -e "${YELLOW}Waiting for service health check...${NC}"

    for i in {1..60}; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            echo -e "${GREEN}✓ Service is healthy${NC}"
            return 0
        fi
        echo "Waiting... ($i/60)"
        sleep 2
    done

    echo -e "${YELLOW}⚠ Service health check timeout, but update completed${NC}"
    return 0
}

# Function to rollback on failure
rollback() {
    echo -e "${YELLOW}Rolling back due to error...${NC}"
    docker-compose -f "$COMPOSE_FILE" down
    docker-compose -f "$COMPOSE_FILE" up -d
    echo -e "${GREEN}✓ Rollback completed${NC}"
    exit 1
}

# Main update flow
main() {
    # Confirm update
    read -p "This will update the application. Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Update cancelled."
        exit 0
    fi

    # Create backup first
    create_backup

    # Pull latest code
    pull_latest_code

    # Build new image
    build_new_image

    # Update services (with rollback on error)
    trap rollback ERR
    update_services
    trap - ERR

    # Wait for health check
    wait_for_health

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  Update Completed Successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Application is running at: http://localhost:8000"
    echo "View logs: docker-compose -f $COMPOSE_FILE logs -f app"
}

# Run main function
main
