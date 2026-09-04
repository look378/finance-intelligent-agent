#!/bin/bash
set -e

echo "🚀 启动金融智能客服..."

# 等待依赖服务就绪
wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    echo "⏳ 等待 $service_name ($host:$port)..."
    for i in {1..60}; do
        if nc -z "$host" "$port" 2>/dev/null; then
            echo "✅ $service_name 已就绪"
            return 0
        fi
        echo "   $service_name 未就绪，重试 ($i/60)"
        sleep 2
    done
    echo "❌ $service_name 在超时时间内未就绪"
    exit 1
}

if [ -n "$DATABASE_HOST" ]; then
    wait_for_service "$DATABASE_HOST" "${DATABASE_PORT:-5432}" "PostgreSQL"
fi
if [ -n "$REDIS_HOST" ]; then
    wait_for_service "$REDIS_HOST" "${REDIS_PORT:-6379}" "Redis"
fi

# 初始化数据库表（幂等：已存在则跳过）
echo "🔧 初始化数据库表..."
python scripts/init_db.py || echo "⚠️ 数据库初始化（demo 数据）跳过，继续启动"

# 启动应用
echo "🎯 启动 uvicorn 服务..."
exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
