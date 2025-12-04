"""
Celery知识抽取任务定义
"""
from celery import Task
from typing import Dict, Any, List
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from celery_app import celery_app
from app.database import SessionLocal
from .services.extraction_service import ExtractionService
from app.config import settings

logger = logging.getLogger(__name__)


class ExtractionTask(Task):
    """抽取任务基类"""

    def __call__(self, *args, **kwargs):
        """重写调用方法，确保每个任务都有数据库会话"""
        with SessionLocal() as db:
            self.db = db
            return self.run(*args, **kwargs)


@celery_app.task(
    bind=True,
    base=ExtractionTask,
    name="app.tasks.extraction_tasks.extract_single_result",
    queue="extraction_queue"
)
def extract_single_result(self, result_id: int) -> Dict[str, Any]:
    """
    抽取单个爬取结果

    Args:
        result_id: 爬取结果ID

    Returns:
        dict: 执行结果
    """
    logger.info(f"Celery抽取任务开始: result_id={result_id}, 任务ID: {self.request.id}")

    try:
        extraction_service = ExtractionService(self.db)

        success, message, extraction_record_id = extraction_service.extract_from_result(result_id)

        if success:
            logger.info(f"✓ 抽取任务完成: result_id={result_id}, {message}")
            return {
                'success': True,
                'result_id': result_id,
                'extraction_record_id': extraction_record_id,
                'message': message
            }
        else:
            logger.error(f"❌ 抽取任务失败: result_id={result_id}, {message}")
            return {
                'success': False,
                'result_id': result_id,
                'extraction_record_id': extraction_record_id,
                'message': message
            }

    except Exception as e:
        logger.error(f"❌ 抽取任务异常: result_id={result_id}, {e}")
        return {
            'success': False,
            'result_id': result_id,
            'message': str(e)
        }


@celery_app.task(
    bind=True,
    base=ExtractionTask,
    name="app.tasks.extraction_tasks.retry_failed_extractions",
    queue="extraction_queue"
)
def retry_failed_extractions(self) -> Dict[str, Any]:
    """
    重试所有失败的抽取

    Returns:
        dict: 执行结果
    """
    logger.info(f"Celery重试失败抽取任务开始: 任务ID: {self.request.id}")

    try:
        extraction_service = ExtractionService(self.db)

        success_count, failed_count = extraction_service.retry_failed_extractions()

        logger.info(f"✓ 重试任务完成: 成功={success_count}, 失败={failed_count}")

        return {
            'success': True,
            'message': '重试完成',
            'success_count': success_count,
            'failed_count': failed_count
        }

    except Exception as e:
        logger.error(f"❌ 重试任务异常: {e}")
        return {
            'success': False,
            'message': str(e)
        }


@celery_app.task(
    bind=True,
    base=ExtractionTask,
    name="app.tasks.extraction_tasks.batch_extract_all_pending_parallel",
    queue="extraction_queue"
)
def batch_extract_all_pending_parallel(self, max_workers: int = None) -> Dict[str, Any]:
    """
    批量并行抽取所有待处理的爬取结果（使用ThreadPoolExecutor + tqdm）

    此任务会：
    1. 获取所有未进行信息抽取的结果
    2. 使用线程池并行处理
    3. 显示实时进度条（通过日志）
    4. 自动处理错误和重试

    Args:
        max_workers: 最大并行线程数，默认从配置读取

    Returns:
        dict: 执行结果，包含成功数、失败数、错误列表等
    """
    logger.info(f"Celery并行批量抽取任务开始: 任务ID: {self.request.id}")

    try:
        extraction_service = ExtractionService(self.db)

        # 获取所有待抽取的结果
        result_ids = extraction_service.get_pending_results(limit=None)

        if not result_ids:
            logger.info("没有待抽取的结果")
            return {
                'success': True,
                'message': '没有待抽取的结果',
                'total': 0,
                'success_count': 0,
                'failed_count': 0,
                'errors': []
            }

        logger.info(f"找到 {len(result_ids)} 个待抽取的结果，准备并行处理")

        # 设置并行线程数
        if max_workers is None:
            max_workers = getattr(settings, 'kg_max_workers', 10)

        success_count = 0
        failed_count = 0
        errors = []

        def process_single_result(result_id: int) -> Dict[str, Any]:
            """处理单个结果的函数"""
            # 每个线程需要自己的数据库会话
            with SessionLocal() as thread_db:
                thread_service = ExtractionService(thread_db)
                success, message, extraction_record_id = thread_service.extract_from_result(result_id)

                return {
                    'result_id': result_id,
                    'success': success,
                    'message': message,
                    'extraction_record_id': extraction_record_id
                }

        # 使用ThreadPoolExecutor + tqdm并行处理
        logger.info(f"启用并行处理，线程数: {max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_result_id = {
                executor.submit(process_single_result, result_id): result_id
                for result_id in result_ids
            }

            # 使用tqdm显示进度条
            logger.info("=" * 80)
            logger.info(f"📊 开始并行处理 {len(result_ids)} 个结果...")
            logger.info("=" * 80)

            # 创建进度条（通过日志输出进度信息）
            with tqdm(
                total=len(result_ids),
                desc="🔄 知识图谱抽取进度",
                unit="docs",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            ) as pbar:
                for future in as_completed(future_to_result_id):
                    result_id = future_to_result_id[future]

                    try:
                        result = future.result()

                        if result['success']:
                            success_count += 1
                            pbar.set_postfix_str(f"✓ {success_count} | ✗ {failed_count}")
                        else:
                            failed_count += 1
                            error_msg = f"result_id={result_id}: {result['message']}"
                            errors.append(error_msg)
                            pbar.set_postfix_str(f"✓ {success_count} | ✗ {failed_count}")
                            logger.error(f"✗ {error_msg}")

                    except Exception as e:
                        failed_count += 1
                        error_msg = f"result_id={result_id}: {str(e)}"
                        errors.append(error_msg)
                        pbar.set_postfix_str(f"✓ {success_count} | ✗ {failed_count}")
                        logger.error(f"✗ 异常: {error_msg}")

                    # 更新进度条
                    pbar.update(1)

        logger.info("=" * 80)
        logger.info(f"✓ 并行批量抽取任务完成")
        logger.info(f"总数: {len(result_ids)} | 成功: {success_count} | 失败: {failed_count}")
        logger.info("=" * 80)

        return {
            'success': True,
            'message': f'并行批量抽取完成',
            'total': len(result_ids),
            'success_count': success_count,
            'failed_count': failed_count,
            'errors': errors[:50],  # 只返回前50个错误
            'max_workers': max_workers
        }

    except Exception as e:
        logger.error(f"❌ 并行批量抽取任务异常: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            'success': False,
            'message': str(e),
            'total': 0,
            'success_count': 0,
            'failed_count': 0,
            'errors': [str(e)]
        }
