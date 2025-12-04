"""
知识图谱抽取服务
集成LLM抽取器，管理抽取流程和状态更新
"""
import logging
import time
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.core.models.extraction import ExtractionRecord, ExtractionStatus
from app.core.models.result import CrawlResult
from .kg_extractor import KnowledgeGraphExtractor
from .neo4j_service import Neo4jService
from app.config import settings

logger = logging.getLogger(__name__)


class ExtractionService:
    """知识图谱抽取服务"""

    def __init__(self, db: Session):
        """初始化抽取服务

        Args:
            db: 数据库会话
        """
        self.db = db
        self.max_retries = settings.kg_max_retries
        self.config_path = settings.kg_config_path

        # 延迟初始化extractor（避免启动时就加载）
        self._extractor = None

    @property
    def extractor(self) -> KnowledgeGraphExtractor:
        """懒加载LLM抽取器"""
        if self._extractor is None:
            logger.info(f"初始化LLM抽取器: {self.config_path}")
            self._extractor = KnowledgeGraphExtractor(self.config_path)
        return self._extractor

    def create_extraction_record(self, result_id: int) -> ExtractionRecord:
        """创建抽取记录（带并发安全检查）

        Args:
            result_id: 爬取结果ID

        Returns:
            创建的抽取记录
        """
        # 先检查是否已存在（防止并发重复创建）
        existing_record = self.db.query(ExtractionRecord).filter(
            ExtractionRecord.result_id == result_id
        ).first()

        if existing_record:
            logger.info(f"抽取记录已存在: ID={existing_record.id}, result_id={result_id}")
            return existing_record

        # 创建新记录（使用try-except处理并发插入冲突）
        try:
            record = ExtractionRecord(
                result_id=result_id,
                status=ExtractionStatus.PENDING
            )
            self.db.add(record)
            self.db.commit()
            self.db.refresh(record)
            logger.info(f"创建抽取记录: ID={record.id}, result_id={result_id}")
            return record
        except IntegrityError:
            # 并发插入冲突，回滚并重新查询
            self.db.rollback()
            existing_record = self.db.query(ExtractionRecord).filter(
                ExtractionRecord.result_id == result_id
            ).first()
            logger.warning(f"并发插入冲突，使用已存在记录: ID={existing_record.id}, result_id={result_id}")
            return existing_record

    def extract_from_result(
        self,
        result_id: int,
        extraction_record_id: Optional[int] = None
    ) -> Tuple[bool, str, Optional[int]]:
        """从爬取结果中抽取知识图谱

        Args:
            result_id: 爬取结果ID
            extraction_record_id: 抽取记录ID（如果已存在）

        Returns:
            (是否成功, 消息, 抽取记录ID)
        """
        # 获取爬取结果
        crawl_result = self.db.query(CrawlResult).filter(
            CrawlResult.id == result_id
        ).first()

        if not crawl_result:
            logger.error(f"爬取结果不存在: result_id={result_id}")
            return False, f"爬取结果不存在: {result_id}", None

        # 检查是否有内容
        if not crawl_result.content or crawl_result.content.strip() == "":
            logger.warning(f"爬取结果无内容: result_id={result_id}")
            return False, "爬取结果无内容", None

        # 获取或创建抽取记录
        if extraction_record_id:
            extraction_record = self.db.query(ExtractionRecord).filter(
                ExtractionRecord.id == extraction_record_id
            ).first()
            if not extraction_record:
                logger.error(f"抽取记录不存在: {extraction_record_id}")
                return False, f"抽取记录不存在: {extraction_record_id}", None
        else:
            extraction_record = self.create_extraction_record(result_id)
            extraction_record_id = extraction_record.id

        # 更新状态为运行中
        extraction_record.status = ExtractionStatus.RUNNING
        extraction_record.extraction_started_at = datetime.utcnow()
        self.db.commit()

        # 执行抽取（带重试）
        for attempt in range(self.max_retries):
            try:
                logger.info(f"开始抽取 (第 {attempt + 1}/{self.max_retries} 次): result_id={result_id}")

                # 调用LLM抽取器
                extraction_result = self.extractor.extract_from_text(crawl_result.content)

                # 获取三元组列表（兼容新旧API）
                triplets = extraction_result.relations if hasattr(extraction_result, 'relations') else extraction_result.triplets

                logger.info(f"抽取完成: 提取到 {len(triplets)} 个三元组")

                # 添加元数据到三元组（保留source_url和doc_id）
                triplets_with_metadata = []
                for triplet in triplets:
                    triplet_dict = triplet.to_dict()
                    triplet_dict['source_url'] = crawl_result.url
                    triplet_dict['doc_id'] = f"crawl_result_{result_id}"
                    triplets_with_metadata.append(triplet_dict)

                # 导入到Neo4j
                success, neo4j_error = self._import_to_neo4j(
                    triplets_with_metadata,
                    extraction_record
                )

                if success:
                    # 更新抽取记录为成功
                    extraction_record.status = ExtractionStatus.SUCCESS
                    extraction_record.triplet_count = len(triplets)
                    extraction_record.extraction_completed_at = datetime.utcnow()
                    extraction_record.error_message = None
                    self.db.commit()

                    logger.info(f"✓ 抽取并导入成功: result_id={result_id}, triplets={len(triplets)}")
                    return True, f"成功抽取 {len(triplets)} 个三元组", extraction_record_id
                else:
                    # Neo4j导入失败，但抽取成功
                    extraction_record.status = ExtractionStatus.SUCCESS
                    extraction_record.triplet_count = len(triplets)
                    extraction_record.extraction_completed_at = datetime.utcnow()
                    extraction_record.neo4j_error_message = neo4j_error
                    self.db.commit()

                    logger.warning(f"⚠ 抽取成功但Neo4j导入失败: {neo4j_error}")
                    return True, f"抽取成功 {len(triplets)} 个三元组，但Neo4j导入失败: {neo4j_error}", extraction_record_id

            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ 抽取失败 (第 {attempt + 1}/{self.max_retries} 次): {error_msg}")

                # 更新重试次数
                extraction_record.retry_count = attempt + 1
                extraction_record.error_message = error_msg
                self.db.commit()

                # 检查是否需要继续重试
                if attempt < self.max_retries - 1:
                    # 计算指数退避等待时间
                    wait_time = min(2 ** attempt, 60)  # 最大等待60秒
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 达到最大重试次数，标记为失败
                    extraction_record.status = ExtractionStatus.FAILED
                    extraction_record.extraction_completed_at = datetime.utcnow()
                    self.db.commit()

                    logger.error(f"已达到最大重试次数，抽取失败: result_id={result_id}")
                    return False, f"抽取失败 (重试{self.max_retries}次): {error_msg}", extraction_record_id

        # 不应该到达这里
        return False, "未知错误", extraction_record_id

    def _import_to_neo4j(
        self,
        triplets: List[Dict[str, Any]],
        extraction_record: ExtractionRecord
    ) -> Tuple[bool, Optional[str]]:
        """导入三元组到Neo4j

        Args:
            triplets: 三元组列表
            extraction_record: 抽取记录

        Returns:
            (是否成功, 错误信息)
        """
        if not triplets:
            logger.info("无三元组需要导入")
            extraction_record.neo4j_imported = True
            extraction_record.neo4j_imported_at = datetime.utcnow()
            self.db.commit()
            return True, None

        try:
            with Neo4jService() as neo4j:
                success_count, failed_count, error_msg = neo4j.import_triplets_with_retry(
                    triplets,
                    max_retries=settings.neo4j_max_retries
                )

                # 更新Neo4j导入状态
                if failed_count == 0:
                    extraction_record.neo4j_imported = True
                    extraction_record.neo4j_imported_at = datetime.utcnow()
                    extraction_record.neo4j_error_message = None
                    self.db.commit()
                    logger.info(f"✓ Neo4j导入成功: {success_count} 个三元组")
                    return True, None
                else:
                    extraction_record.neo4j_imported = False
                    extraction_record.neo4j_error_message = error_msg
                    extraction_record.neo4j_retry_count += 1
                    self.db.commit()
                    logger.error(f"❌ Neo4j导入失败: {error_msg}")
                    return False, error_msg

        except Exception as e:
            error_msg = f"Neo4j导入异常: {e}"
            extraction_record.neo4j_imported = False
            extraction_record.neo4j_error_message = error_msg
            extraction_record.neo4j_retry_count += 1
            self.db.commit()
            logger.error(f"❌ {error_msg}")
            return False, error_msg

    def get_pending_results(self, limit: int = None) -> List[int]:
        """获取待抽取的爬取结果ID列表

        Args:
            limit: 最大数量限制

        Returns:
            result_id列表
        """
        # 查询所有有内容但未抽取的爬取结果
        query = self.db.query(CrawlResult.id).filter(
            CrawlResult.content.isnot(None),
            CrawlResult.content != "",
            ~CrawlResult.id.in_(
                self.db.query(ExtractionRecord.result_id).filter(
                    ExtractionRecord.status == ExtractionStatus.SUCCESS
                )
            )
        )

        if limit:
            query = query.limit(limit)

        result_ids = [row[0] for row in query.all()]
        logger.info(f"找到 {len(result_ids)} 个待抽取的结果")
        return result_ids

    def retry_failed_extractions(self) -> Tuple[int, int]:
        """重试所有失败的抽取

        Returns:
            (成功数量, 失败数量)
        """
        # 查询所有失败的抽取记录
        failed_records = self.db.query(ExtractionRecord).filter(
            ExtractionRecord.status == ExtractionStatus.FAILED
        ).all()

        logger.info(f"找到 {len(failed_records)} 个失败的抽取记录")

        success_count = 0
        failed_count = 0

        for record in failed_records:
            logger.info(f"重试抽取: extraction_record_id={record.id}, result_id={record.result_id}")

            # 重置重试次数
            record.retry_count = 0
            record.status = ExtractionStatus.PENDING
            self.db.commit()

            success, message, _ = self.extract_from_result(
                record.result_id,
                extraction_record_id=record.id
            )

            if success:
                success_count += 1
            else:
                failed_count += 1

        logger.info(f"重试完成: 成功 {success_count}, 失败 {failed_count}")
        return success_count, failed_count
