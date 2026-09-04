#!/usr/bin/env python
"""
企业知识库批量导入脚本。

将指定目录下的文档（PDF/TXT/MD）批量导入 RAG 知识库：
1. 扫描目录（支持递归子目录，子目录名作为 category 元数据）
2. 支持 sidecar 元数据文件（同名 .json，可标注 title/category/tags/version/effective_date）
3. 预处理 → 分块 → 向量化 → 写入 Qdrant
4. 可选的文档去重（按内容哈希，跳过已导入内容）

用法（从项目根目录）：
    python scripts/import_kb.py --dir ./kb/finance --category 产品条款
    python scripts/import_kb.py --dir ./kb --recursive --dedupe
    python scripts/import_kb.py --dir ./kb --strategy recursive --chunk-size 600 --overlap 80

依赖：项目依赖已安装（app.services.*），embedding 模型已就绪（models/bge-m3 或自动下载）。
"""
import argparse
import asyncio
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

# 允许作为独立脚本从项目根运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.documents.chunking import (
    FixedSizeChunking,
    SemanticChunking,
    RecursiveCharacterChunking,
)
from app.services.documents.preprocessing import DocumentPreprocessor
from app.services.documents.base import Document
from app.services.embeddings import EmbeddingFactory
from app.services.retrieval.qdrant_client import QdrantClient

CHUNK_STRATEGIES = {
    "fixed": FixedSizeChunking,
    "semantic": SemanticChunking,
    "recursive": RecursiveCharacterChunking,
}

SUPPORTED_EXT = {".txt", ".md", ".pdf"}


def _read_text_file(path: Path) -> str:
    """Read text file with UTF-8 fallback."""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="latin-1")


def _read_pdf(path: Path) -> str:
    """Extract text from PDF."""
    import pypdf

    text = []
    with open(path, "rb") as f:
        reader = pypdf.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)


def _load_sidecar_metadata(path: Path) -> dict:
    """Load optional sidecar metadata (same-name .json)."""
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"  ⚠️ 忽略无效的 sidecar 元数据: {sidecar.name}")
    return {}


def _content_hash(text: str) -> str:
    """Compute content hash for dedup."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect_files(root: Path, recursive: bool) -> list:
    """Collect supported files, optionally recursive."""
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    else:
        files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    # 排除 sidecar json（同名 .json 不作为独立文档）
    return sorted(files)


def _read_document(path: Path) -> tuple:
    """Read document content by extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    return _read_text_file(path)


async def main() -> None:
    parser = argparse.ArgumentParser(description="企业知识库批量导入脚本")
    parser.add_argument("--dir", required=True, help="文档目录")
    parser.add_argument("--recursive", action="store_true", help="递归扫描子目录（子目录名作为 category）")
    parser.add_argument("--strategy", default="semantic", choices=list(CHUNK_STRATEGIES), help="分块策略")
    parser.add_argument("--chunk-size", type=int, default=512, help="分块大小（字符数）")
    parser.add_argument("--overlap", type=int, default=50, help="分块重叠（字符数）")
    parser.add_argument("--category", default="", help="默认分类（覆盖子目录推断）")
    parser.add_argument("--dedupe", action="store_true", help="按内容哈希去重（跳过已导入内容）")
    parser.add_argument("--qdrant-url", default="http://localhost:6333", help="Qdrant URL")
    parser.add_argument("--collection", default="documents", help="Qdrant collection 名")
    parser.add_argument("--max-chars", type=int, default=100000, help="单文档最大字符数")
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        print(f"❌ 目录不存在: {root}")
        sys.exit(1)

    files = _collect_files(root, args.recursive)
    if not files:
        print(f"⚠️ 目录下未找到支持的文档（{', '.join(SUPPORTED_EXT)}）: {root}")
        return

    print(f"📁 扫描到 {len(files)} 个文档")

    # 初始化组件
    embedding_service = EmbeddingFactory.create_from_settings(use_cache=False)
    qdrant = QdrantClient(
        url=args.qdrant_url,
        collection_name=args.collection,
        embedding_service=embedding_service,
    )

    # 确保 collection 存在
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import Distance, VectorParams

    qc = AsyncQdrantClient(url=args.qdrant_url)
    collections = await qc.get_collections()
    existing = {c.name for c in collections.collections}
    if args.collection not in existing:
        await qc.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=embedding_service.dimensions, distance=Distance.COSINE),
        )
        print(f"🆕 创建 collection '{args.collection}' (dim={embedding_service.dimensions})")
    await qc.close()

    chunker = CHUNK_STRATEGIES[args.strategy]()
    preprocessor = DocumentPreprocessor(max_length=args.max_chars)

    imported = 0
    skipped = 0
    failed = 0
    seen_hashes = set()

    for path in files:
        print(f"\n📄 {path.relative_to(root)}")
        try:
            text = _read_document(path)
            if not text.strip():
                print("  ⚠️ 空文档，跳过")
                skipped += 1
                continue

            # 去重
            if args.dedupe:
                h = _content_hash(text)
                if h in seen_hashes:
                    print("  ⚠️ 内容重复，跳过")
                    skipped += 1
                    continue
                seen_hashes.add(h)

            # 元数据：sidecar > 子目录推断 > --category 默认
            sidecar_meta = _load_sidecar_metadata(path)
            doc_meta = dict(sidecar_meta)
            if not doc_meta.get("category"):
                if args.category:
                    doc_meta["category"] = args.category
                elif args.recursive and path.parent != root:
                    doc_meta["category"] = path.parent.name
                else:
                    doc_meta["category"] = "未分类"
            doc_meta.setdefault("source", "enterprise_kb")
            doc_meta.setdefault("original_filename", path.name)
            doc_meta["file_type"] = path.suffix.lower().lstrip(".")

            # 预处理
            processed, enhanced = await preprocessor.process(text, doc_meta)
            doc_meta.update(enhanced)

            # 分块
            document = Document(
                document_id=str(uuid.uuid4()),
                title=doc_meta.get("title") or path.stem,
                content=processed,
                file_type=doc_meta["file_type"],
                metadata=doc_meta,
            )
            chunks = await chunker.chunk(
                document=document,
                max_chunk_size=args.chunk_size,
                chunk_overlap=args.overlap,
            )
            chunk_texts = [c.content for c in chunks]
            embedding_result = await embedding_service.embed(chunk_texts)

            # 写入 Qdrant
            payloads = [
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "index": c.index,
                    **c.metadata,
                }
                for c in chunks
            ]
            await qdrant.add(
                ids=[c.chunk_id for c in chunks],
                vectors=embedding_result.embeddings,
                payloads=payloads,
            )
            imported += 1
            print(f"  ✅ 已导入：{len(chunks)} 个分块 (category={doc_meta['category']})")

        except Exception as e:
            failed += 1
            print(f"  ❌ 导入失败: {type(e).__name__}: {str(e)[:120]}")

    print("\n" + "=" * 50)
    print(f"📊 导入完成：成功 {imported}，跳过 {skipped}，失败 {failed}")
    print(f"💡 提示：可调用 GET /api/v1/documents/health 或在聊天中验证知识问答")


if __name__ == "__main__":
    asyncio.run(main())
