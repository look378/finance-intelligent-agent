#!/bin/bash
# ============================================================
# 金融智能客服 一键部署脚本
#
# 用法（在开发者本地执行）：
#   1. 打包:  ./scripts/deploy.sh build
#      生成 dist/finance-agent.tar.gz（含代码 + 部署配置 + 模型打包说明）
#   2. 上传:  scp dist/finance-agent.tar.gz user@server:/opt/finance-agent/
#   3. 服务器: 解压后 ./scripts/deploy.sh install  或手动 docker compose 启动
#
# 服务器启动（手动方式）：
#   cd /opt/finance-agent
#   cp .env.production.example .env.production   # 填 DEEPSEEK_API_KEY / SECRET_KEY
#   docker compose -f docker-compose.prod.yml up -d --build
#   docker compose -f docker-compose.prod.yml exec app python scripts/init_kb.py
# ============================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
OUTPUT_TARBALL="$DIST_DIR/finance-agent.tar.gz"

build() {
    echo "📦 构建部署包..."
    mkdir -p "$DIST_DIR"

    # 打包：代码 + 部署配置 + 模型说明
    # 注意：models/bge-m3（2.3GB）不自动打入，见下方说明
    tar -czf "$OUTPUT_TARBALL" \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache' \
        --exclude='pytest-cache-files-*' \
        --exclude='.modelscope-*' \
        --exclude='docs/server.log' \
        app/ scripts/ migrations/ deploy/ deployment/ \
        docker-compose.yml docker-compose.prod.yml docker-compose.fullstack.yml \
        Dockerfile .dockerignore \
        requirements.txt requirements-dev.txt pyproject.toml alembic.ini \
        .env.production.example README.md 2>/dev/null

    echo "✅ 部署包已生成: $OUTPUT_TARBALL"
    echo ""
    echo "⚠️  注意：本地 embedding 模型（models/bge-m3，约2.3GB）未包含在包内。"
    echo "    服务器上二选一："
    echo "      1) 手动拷贝:  scp -r models user@server:/opt/finance-agent/"
    echo "      2) 服务器联网:  docker compose ... exec app python scripts/init_kb.py（自动下载/提示）"
}

install() {
    echo "🚀 开始安装（需在服务器上执行，且已解压部署包）..."
    echo ""
    echo "步骤 1：准备环境配置"
    echo "  cp .env.production.example .env.production"
    echo "  编辑 .env.production，至少填写："
    echo "    DEEPSEEK_API_KEY=sk-xxxx        # 必填"
    echo "    SECRET_KEY=随机32+字符            # 必填"
    echo "    POSTGRES_PASSWORD=强密码         # 建议修改"
    echo "    REDIS_PASSWORD=强密码            # 建议修改"
    echo ""
    echo "步骤 2：准备 embedding 模型"
    echo "  将 models/bge-m3 目录放到 ./models/ 下（或联网执行知识库初始化）"
    echo ""
    echo "步骤 3：启动服务"
    echo "  docker compose -f docker-compose.prod.yml up -d --build"
    echo ""
    echo "步骤 4：初始化知识库（首次）"
    echo "  docker compose -f docker-compose.prod.yml exec app python scripts/init_kb.py"
    echo ""
    echo "步骤 5：验证"
    echo "  curl http://localhost:8000/health"
    echo "  浏览器打开 http://<服务器IP>:8000/  即聊天页面"
}

case "${1:-}" in
    build) build ;;
    install) install ;;
    *)
        echo "用法: $0 {build|install}"
        echo "  build   - 本地打包部署包（dist/finance-agent.tar.gz）"
        echo "  install - 打印服务器安装步骤（在服务器上执行）"
        exit 1
        ;;
esac
