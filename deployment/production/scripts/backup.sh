#!/bin/bash
# ==========================================
# 金融智能客服 Database Backup Script
# ==========================================
# This script creates automated backups of the PostgreSQL database
# Usage: ./backup.sh [retention_days]

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="$PROJECT_ROOT/backups"
RETENTION_DAYS=${1:-7}
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  金融智能客服 Database Backup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# Generate backup filename with timestamp
BACKUP_FILE="$BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql"

echo "Creating database backup..."
echo "Backup file: $BACKUP_FILE"
echo ""

# Backup database
if docker-compose -f "$COMPOSE_FILE" exec -T db pg_dump -U postgres finance_agent > "$BACKUP_FILE" 2>/dev/null; then
    # Compress backup
    gzip "$BACKUP_FILE"
    BACKUP_FILE="${BACKUP_FILE}.gz"

    echo -e "${GREEN}✓ Database backup created successfully${NC}"
    echo "Compressed backup: $BACKUP_FILE"
    echo "Size: $(du -h "$BACKUP_FILE" | cut -f1)"
else
    echo -e "${RED}✗ Database backup failed${NC}"
    exit 1
fi

echo ""
echo "Cleaning up old backups (keeping last $RETENTION_DAYS days)..."

# Remove old backups
find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -type f -mtime +$RETENTION_DAYS -delete

OLD_BACKUPS=$(find "$BACKUP_DIR" -name "db_backup_*.sql.gz" -type f | wc -l)
echo "Remaining backups: $OLD_BACKUPS"

echo ""
echo -e "${GREEN}Backup completed successfully!${NC}"
