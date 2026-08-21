"""数据库层（M14）：SQLAlchemy 引擎 + SQLite WAL + 会话管理。

- 本地默认 SQLite，公网可通过 DATABASE_URL 切 Postgres（SQLAlchemy 抽象，切换无缝）；
- SQLite 启用 ``PRAGMA journal_mode=WAL`` + ``busy_timeout``，防评审并发写锁致 500；
- 四表结构见 models.py（users / sessions / messages / plans_cache）。
"""
from __future__ import annotations

import logging
from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATABASE_URL

logger = logging.getLogger("healthpick.database")


class Base(DeclarativeBase):
    """ORM 声明基类。"""


def _build_engine() -> Engine:
    """按 DATABASE_URL 构造引擎；SQLite 走 WAL + 忙等待，防并发写锁。"""
    is_sqlite = DATABASE_URL.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}
    engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)

    if is_sqlite:

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _conn_record) -> None:  # noqa: ANN001
            cursor = dbapi_conn.cursor()
            try:
                # WAL：读写不互斥，防多并发下 database is locked
                cursor.execute("PRAGMA journal_mode=WAL")
                # 忙等待 5s：短冲突自旋而非直接报错
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.execute("PRAGMA foreign_keys=ON")
            finally:
                cursor.close()

    return engine


engine: Engine = _build_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    """建表（幂等）。应用启动时调用。"""
    from app import models  # noqa: F401  # 确保四表已注册到 Base.metadata

    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith("sqlite"):
        with engine.connect() as conn:
            mode = conn.exec_driver_sql("PRAGMA journal_mode").scalar()
        logger.info("SQLite journal_mode=%s（期望 wal）", mode)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖：请求级数据库会话。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
