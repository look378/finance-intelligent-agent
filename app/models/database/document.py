"""Document database model"""
from sqlalchemy import String, Text, Integer, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database.base import Base, TimestampMixin


class Document(Base, TimestampMixin):
    """
    Document model for tracking ingested documents.

    Note: Actual document content and embeddings are stored in the
    external vector database service. This model only stores metadata.

    Attributes:
        id: Primary key
        external_doc_id: Document ID in vector service (must be unique)
        title: Document title
        source: Source file path or URL
        doc_type: Document type (pdf, txt, html, etc.)
        chunk_count: Number of chunks in vector database
        metadata: Additional metadata as JSON
        is_active: Whether document is active
    """
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    external_doc_id: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    source: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    doc_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )
    doc_metadata: Mapped[dict | None] = mapped_column(JSON, default=None)
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, external_id={self.external_doc_id}, title={self.title})>"
