"""
知识图谱抽取记录模型
记录每个爬取结果的信息抽取状态
"""
from datetime import datetime
from enum import Enum
from sqlalchemy import Column, Integer, String, DateTime, Text, Enum as SQLEnum, Boolean, ForeignKey
from app.database import Base


class ExtractionStatus(str, Enum):
    """抽取状态枚举"""
    PENDING = "pending"       # 待抽取
    RUNNING = "running"       # 抽取中
    SUCCESS = "success"       # 抽取成功
    FAILED = "failed"         # 抽取失败


class ExtractionRecord(Base):
    """知识图谱抽取记录表"""
    __tablename__ = "extraction_records"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # 关联爬取结果（每个result_id只允许一个成功的抽取记录）
    result_id = Column(Integer, ForeignKey("crawl_results.id"), nullable=False, unique=True, index=True, comment="关联爬取结果ID（唯一）")

    # 抽取状态
    status = Column(
        SQLEnum(ExtractionStatus),
        default=ExtractionStatus.PENDING,
        nullable=False,
        index=True,
        comment="抽取状态"
    )

    # 时间信息
    extraction_started_at = Column(DateTime, nullable=True, comment="开始抽取时间")
    extraction_completed_at = Column(DateTime, nullable=True, comment="完成抽取时间")
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, comment="记录创建时间")

    # 结果统计
    triplet_count = Column(Integer, default=0, comment="抽取到的三元组数量")

    # 错误信息
    error_message = Column(Text, nullable=True, comment="错误信息")
    retry_count = Column(Integer, default=0, comment="重试次数")

    # Neo4j导入状态
    neo4j_imported = Column(Boolean, default=False, index=True, comment="是否已导入Neo4j")
    neo4j_imported_at = Column(DateTime, nullable=True, comment="Neo4j导入时间")
    neo4j_error_message = Column(Text, nullable=True, comment="Neo4j导入错误信息")
    neo4j_retry_count = Column(Integer, default=0, comment="Neo4j导入重试次数")

    def __repr__(self):
        return f"<ExtractionRecord(id={self.id}, result_id={self.result_id}, status={self.status})>"
