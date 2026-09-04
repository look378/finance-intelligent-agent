#!/bin/bash
# ==========================================
# 金融智能客服 Health Monitor Script
# ==========================================
# This script monitors the health of all services
# Usage: ./monitor.sh [continuous]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
CONTINUOUS=${1:-false}

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to check service health
check_service() {
    local service=$1
    local name=$2

    if docker-compose -f "$COMPOSE_FILE" ps "$service" | grep -q "Up"; then
        echo -e "${GREEN}✓${NC} $name is running"
        return 0
    else
        echo -e "${RED}✗${NC} $name is not running"
        return 1
    fi
}

# Function to check API health
check_api() {
    if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} API is healthy"
        return 0
    else
        echo -e "${RED}✗${NC} API is not responding"
        return 1
    fi
}

# Function to get service stats
get_stats() {
    echo ""
    echo -e "${BLUE}Service Statistics:${NC}"
    echo ""

    # Container stats
    echo "Container Status:"
    docker-compose -f "$COMPOSE_FILE" ps
    echo ""

    # Resource usage
    echo "Resource Usage:"
    docker stats --no-stream $(docker-compose -f "$COMPOSE_FILE" ps -q) 2>/dev/null || echo "Stats not available"
    echo ""

    # Disk usage
    echo "Disk Usage:"
    docker system df
    echo ""
}

# Main monitor function
monitor() {
    if [ "$CONTINUOUS" = "continuous" ]; then
        echo -e "${BLUE}Starting continuous monitoring (Ctrl+C to stop)${NC}"
        echo ""
        while true; do
            clear
            echo -e "${BLUE}========================================${NC}"
            echo -e "${BLUE}  金融智能客服 Health Monitor${NC}"
            echo -e "${BLUE}========================================${NC}"
            echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
            echo ""

            check_service app "API Service"
            check_service db "PostgreSQL"
            check_service redis "Redis"
            check_service qdrant "Qdrant"
            check_api

            get_stats

            sleep 5
        done
    else
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}  金融智能客服 Health Monitor${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""

        check_service app "API Service"
        check_service db "PostgreSQL"
        check_service redis "Redis"
        check_service qdrant "Qdrant"
        check_api

        get_stats

        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Health Check Complete${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "For continuous monitoring, run: $0 continuous"
    fi
}

# Run monitor
monitor
