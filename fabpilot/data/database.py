"""数据库 engine / session 管理。

业务数据库 URL 完全可配置。生产环境使用 PostgreSQL（见 docker-compose.yml）；测试和本地快速运行可指向 SQLite。
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from fabpilot.config import Settings, get_settings
from fabpilot.data.models import Base


def make_engine(url: str) -> Engine:
    kwargs: dict = {"echo": False, "future": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


_engine: Engine | None = None
_session_factory: sessionmaker | None = None


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        url = get_settings().database_url
        _engine = make_engine(url)
        _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_session_factory() -> sessionmaker:
    get_engine()
    assert _session_factory is not None
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """绑定到全局 engine 的事务性 session scope。"""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def build_session_scope(url: str, create_tables: bool = False):
    """返回一个绑定到 ``url`` 专用 engine 的 session scope。

    供 runtime 与评估层使用，使自定义数据库 URL（例如临时 SQLite 文件）无需触碰全局 engine 即可端到端生效。
    """
    engine = make_engine(url)
    if create_tables:
        Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    @contextmanager
    def _scope() -> Iterator[Session]:
        session = factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    return _scope


def create_all() -> None:
    Base.metadata.create_all(get_engine())


def drop_all() -> None:
    Base.metadata.drop_all(get_engine())
