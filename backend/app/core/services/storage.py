"""
MySQL存储服务
替代原有的JSONStorage，将爬取结果保存到MySQL数据库
"""
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import json
import logging

from app.core.models.task import CrawlTask, TaskStatus
from app.core.models.result import CrawlResult
from .deduplication import DeduplicationService

logger = logging.getLogger(__name__)


class MySQLStorageService:
    """MySQL存储服务"""

    def __init__(self, db: Session):
        self.db = db
        self.dedup_service = DeduplicationService(db)

    def create_task(
        self,
        task_name: str,
        config_file: str,
        celery_task_id: Optional[str] = None
    ) -> CrawlTask:
        """
        创建新的爬取任务

        Args:
            task_name: 任务名称
            config_file: 配置文件路径
            celery_task_id: Celery任务ID

        Returns:
            CrawlTask: 创建的任务对象
        """
        task = CrawlTask(
            task_name=task_name,
            config_file=config_file,
            status=TaskStatus.PENDING,
            celery_task_id=celery_task_id,
            created_at=datetime.utcnow()
        )
        self.db.add(task)
        self.db.commit()
        self.db.refresh(task)

        logger.info(f"创建任务: {task_name} (ID: {task.id})")
        return task

    def update_task_status(
        self,
        task_id: int,
        status: TaskStatus,
        error_message: Optional[str] = None
    ) -> CrawlTask:
        """
        更新任务状态

        Args:
            task_id: 任务ID
            status: 新状态
            error_message: 错误信息（可选）

        Returns:
            CrawlTask: 更新后的任务对象
        """
        task = self.db.query(CrawlTask).filter(CrawlTask.id == task_id).first()
        if not task:
            raise ValueError(f"任务不存在: {task_id}")

        task.status = status

        if status == TaskStatus.RUNNING:
            task.started_at = datetime.utcnow()
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            task.completed_at = datetime.utcnow()

        if error_message:
            task.error_message = error_message

        self.db.commit()
        self.db.refresh(task)

        logger.info(f"更新任务状态: {task_id} -> {status}")
        return task

    def save_results(
        self,
        task_id: int,
        results: List[Dict[str, Any]],
        enable_dedup: bool = True
    ) -> Dict[str, int]:
        """
        保存爬取结果到数据库

        Args:
            task_id: 关联的任务ID
            results: 爬取结果列表（字典格式）
            enable_dedup: 是否启用去重

        Returns:
            dict: 统计信息 {saved: 保存数量, duplicates: 重复数量}
        """
        saved_count = 0
        duplicate_count = 0

        for result_data in results:
            url = result_data.get('url')
            if not url:
                logger.warning("结果缺少URL，跳过")
                continue

            # 去重检查
            is_duplicate = False
            if enable_dedup:
                is_duplicate = self.dedup_service.is_url_duplicate(url)

            # 提取内容字段
            content_dict = result_data.get('content', {})

            # title字段可能是字符串、列表或其他类型，需要统一处理
            raw_title = content_dict.get('title', '')
            if isinstance(raw_title, list):
                # 如果是列表，取第一个元素或拼接
                title = raw_title[0] if raw_title else ''
            elif isinstance(raw_title, dict):
                # 如果是字典，转换为JSON字符串
                title = json.dumps(raw_title, ensure_ascii=False)
            else:
                # 其他情况转换为字符串
                title = str(raw_title) if raw_title else ''

            # content字段可能是字符串、列表或其他类型，需要统一处理
            raw_content = content_dict.get('content', '')
            if isinstance(raw_content, list):
                # 如果是列表，转换为JSON字符串
                content = json.dumps(raw_content, ensure_ascii=False)
            elif isinstance(raw_content, dict):
                # 如果是字典，转换为JSON字符串
                content = json.dumps(raw_content, ensure_ascii=False)
            else:
                # 其他情况转换为字符串
                content = str(raw_content) if raw_content else ''

            # 计算哈希
            url_hash = CrawlResult.compute_url_hash(url)
            content_hash = CrawlResult.compute_content_hash(content) if content else None

            # 提取按钮信息
            button_info = result_data.get('button_info', {})

            # 创建结果记录
            crawl_result = CrawlResult(
                task_id=task_id,
                url=url,
                url_hash=url_hash,
                original_url=result_data.get('original_url'),
                title=title,
                content=content,
                content_hash=content_hash,
                timestamp=datetime.fromisoformat(result_data.get('timestamp', datetime.utcnow().isoformat())),
                new_tab=result_data.get('new_tab', False),
                strategy_used=result_data.get('strategy_used'),
                button_text=button_info.get('text'),
                button_selector=button_info.get('selector'),
                button_href=button_info.get('href'),
                extracted_data=json.dumps(content_dict, ensure_ascii=False),
                is_duplicate=is_duplicate
            )

            self.db.add(crawl_result)

            if is_duplicate:
                duplicate_count += 1
            else:
                saved_count += 1
                # 标记URL已访问
                if enable_dedup:
                    self.dedup_service.mark_url_seen(url)

        # 批量提交
        self.db.commit()

        # 更新任务统计
        task = self.db.query(CrawlTask).filter(CrawlTask.id == task_id).first()
        if task:
            task.total_urls = len(results)
            task.success_count = saved_count
            task.duplicate_count = duplicate_count
            self.db.commit()

        logger.info(f"保存结果: 任务{task_id}, 成功{saved_count}, 重复{duplicate_count}")

        return {
            "saved": saved_count,
            "duplicates": duplicate_count,
            "total": len(results)
        }

    def get_task_results(
        self,
        task_id: int,
        include_duplicates: bool = False
    ) -> List[CrawlResult]:
        """
        获取任务的所有结果

        Args:
            task_id: 任务ID
            include_duplicates: 是否包含重复结果

        Returns:
            List[CrawlResult]: 结果列表
        """
        query = self.db.query(CrawlResult).filter(CrawlResult.task_id == task_id)

        if not include_duplicates:
            query = query.filter(CrawlResult.is_duplicate == False)

        return query.all()

    def get_task_by_id(self, task_id: int) -> Optional[CrawlTask]:
        """获取任务详情"""
        return self.db.query(CrawlTask).filter(CrawlTask.id == task_id).first()
