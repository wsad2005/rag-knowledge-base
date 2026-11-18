"""ORM 模型统一导出。

注意：所有模型必须在此处导入，否则 ``Base.metadata.create_all()`` 建表时会漏表。
"""

from app.models.base import Base, new_id, utcnow
from app.models.chunk import Chunk
from app.models.document import Document, DocumentStatus
from app.models.query_log import QueryLog

__all__ = ["Base", "Chunk", "Document", "DocumentStatus", "QueryLog", "new_id", "utcnow"]
