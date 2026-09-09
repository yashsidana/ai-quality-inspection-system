"""
Database Connection & Engine Factory.
Provides seamless dual-mode connectivity:
- Production: PostgreSQL (via DATABASE_URL env var or Streamlit Secrets)
- Local / Streamlit Cloud default: SQLite (zero-config execution)
"""

import os
from contextlib import contextmanager
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .models import Base

# Read database URL from environment variable
# e.g., postgresql://postgres:password@localhost:5432/quality_inspection
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Default to SQLite local database file
    DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quality_inspection.db")
    DATABASE_URL = f"sqlite:///{DB_PATH}"
    # SQLite requires check_same_thread=False for multithreaded FastAPI / Streamlit usage
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False}
    )
    print(f"[Database] Connected to SQLite local storage: {DB_PATH}")
else:
    # Convert postgres:// to postgresql:// if needed for newer SQLAlchemy versions
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    print(f"[Database] Connected to PostgreSQL host: {DATABASE_URL.split('@')[-1] if '@' in DATABASE_URL else 'remote'}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes all table schemas in the connected database."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_db():
    """Context manager for obtaining database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
