"""数据库连接与会话管理（SQLAlchemy 2.x）。

默认 SQLite（零配置、单文件、便于演示），通过 ``DATABASE_URL`` 可切换到 PostgreSQL。
会话使用 ``expire_on_commit=False``，避免提交后再访问对象属性触发额外查询。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.base import Base

logger = get_logger(__name__)


class Database:
    """引擎 + 会话工厂的薄封装。"""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        url = settings.resolved_database_url
        connect_args: dict[str, object] = {}
        if settings.is_sqlite:
            # FastAPI 会把同步逻辑放到线程池执行，SQLite 连接需要允许跨线程复用
            connect_args["check_same_thread"] = False
            connect_args["timeout"] = 15

        self.engine: Engine = create_engine(
            url,
            echo=settings.database_echo,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        if settings.is_sqlite:
            self._enable_sqlite_pragmas(self.engine)

        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    @staticmethod
    def _enable_sqlite_pragmas(engine: Engine) -> None:
        """WAL 提升并发读性能；外键约束保证删除文档时级联清理 chunk。"""

        @event.listens_for(engine, "connect")
        def _set_pragma(dbapi_connection, _connection_record) -> None:  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

    def create_all(self) -> None:
        Base.metadata.create_all(bind=self.engine)
        logger.info("数据库表结构已就绪：%s", self.engine.url.database)

    @contextmanager
    def session(self) -> Iterator[Session]:
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()
