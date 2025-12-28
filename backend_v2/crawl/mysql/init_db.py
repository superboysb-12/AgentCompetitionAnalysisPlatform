#!/usr/bin/env python3
"""
数据库自动初始化工具
自动创建爬虫所需的数据库表结构

可以作为独立脚本运行，也可以导入到其他模块使用（如 FastAPI lifespan）
"""

import sys
from pathlib import Path

# 添加项目路径到 Python 路径
backend_v2_root = Path(__file__).resolve().parent.parent.parent
crawl_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_v2_root))
sys.path.insert(0, str(crawl_root))

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import OperationalError, ProgrammingError
import logging

logger = logging.getLogger(__name__)


def ensure_tables_exist(mysql_config=None, echo=False):
    """
    确保数据库表存在（如果已存在则跳过）

    这是主要的可复用函数，可在 FastAPI lifespan 或其他地方调用

    Args:
        mysql_config: MySQL 配置字典，如果为 None 则从 backend_v2.config 导入
        echo: 是否显示 SQL 语句，默认 False

    Returns:
        dict: 包含执行结果的字典
            {
                'success': bool,
                'created': bool,  # True 表示新建，False 表示跳过
                'message': str
            }
    """
    try:
        # 导入配置
        if mysql_config is None:
            from settings import MYSQL_CONFIG
            mysql_config = MYSQL_CONFIG

        # 直接导入同目录的 models 模块（不触发 __init__.py）
        import models
        Base = models.Base
        CrawlResultModel = models.CrawlResultModel
        CrawlTaskModel = models.CrawlTaskModel

        # 第一步：创建数据库（如果不存在）
        _ensure_database_exists(mysql_config)

        # 第二步：检查表是否存在
        connection_string = _build_connection_string(mysql_config)
        engine = create_engine(
            connection_string,
            pool_pre_ping=True,
            echo=echo
        )

        # 使用 inspect 检查表是否存在
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        tables_to_check = ['crawl_results', 'crawl_tasks']
        all_exist = all(table in existing_tables for table in tables_to_check)

        if all_exist:
            logger.info(f"✓ 所有表已存在: {', '.join(tables_to_check)}，跳过建表")
            engine.dispose()
            return {
                'success': True,
                'created': False,
                'message': "表已存在，跳过建表"
            }

        # 第三步：创建表（表不存在时）
        logger.info("表不存在，开始创建...")
        Base.metadata.create_all(engine)

        # 验证创建结果
        inspector = inspect(engine)
        created_tables = [t for t in ['crawl_results', 'crawl_tasks'] if t in inspector.get_table_names()]

        if len(created_tables) == 2:
            logger.info(f"✓ 所有表创建成功: {', '.join(created_tables)}")
            engine.dispose()
            return {
                'success': True,
                'created': True,
                'message': "表创建成功"
            }
        else:
            engine.dispose()
            return {
                'success': False,
                'created': False,
                'message': f"部分表创建失败，已创建: {created_tables}"
            }

    except ImportError as e:
        logger.error(f"✗ 导入错误: {e}")
        return {
            'success': False,
            'created': False,
            'message': f"导入错误: {e}"
        }
    except OperationalError as e:
        logger.error(f"✗ 数据库连接失败: {e}")
        return {
            'success': False,
            'created': False,
            'message': f"数据库连接失败: {e}"
        }
    except Exception as e:
        logger.error(f"✗ 初始化失败: {e}")
        return {
            'success': False,
            'created': False,
            'message': f"初始化失败: {e}"
        }


def init_database(verbose=True):
    """
    初始化数据库（兼容旧接口，用于命令行调用）

    Args:
        verbose: 是否显示详细信息

    Returns:
        bool: 是否成功
    """
    try:
        from settings import MYSQL_CONFIG

        if verbose:
            logger.info("正在加载配置...")
            logger.info(f"数据库配置: {MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}")

        # 调用新的封装函数
        result = ensure_tables_exist(MYSQL_CONFIG, echo=verbose)

        if result['success']:
            if verbose:
                logger.info("\n" + "="*60)
                logger.info("数据库初始化完成！")
                logger.info("="*60)
                logger.info(f"数据库: {MYSQL_CONFIG['database']}")
                logger.info("表: crawl_results, crawl_tasks")
                if result['created']:
                    logger.info("状态: 新建")
                else:
                    logger.info("状态: 已存在（跳过）")
                logger.info("="*60 + "\n")
            return True
        else:
            if verbose:
                logger.error(f"初始化失败: {result['message']}")
            return False

    except Exception as e:
        if verbose:
            logger.error(f"✗ 初始化失败: {e}")
            import traceback
            traceback.print_exc()
        return False


def _build_connection_string(mysql_config):
    """构建 MySQL 连接字符串"""
    return (
        f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}"
        f"@{mysql_config['host']}:{mysql_config['port']}/{mysql_config['database']}"
        f"?charset={mysql_config.get('charset', 'utf8mb4')}"
    )


def _ensure_database_exists(mysql_config):
    """确保数据库存在（如果不存在则创建）"""
    try:
        # 连接到 MySQL 服务器（不指定数据库）
        connection_string = (
            f"mysql+pymysql://{mysql_config['user']}:{mysql_config['password']}"
            f"@{mysql_config['host']}:{mysql_config['port']}"
            f"?charset={mysql_config.get('charset', 'utf8mb4')}"
        )

        engine = create_engine(connection_string, pool_pre_ping=True)

        with engine.connect() as conn:
            # 检查数据库是否存在
            result = conn.execute(
                text(f"SHOW DATABASES LIKE '{mysql_config['database']}'")
            )

            if result.fetchone():
                logger.info(f"✓ 数据库 '{mysql_config['database']}' 已存在")
            else:
                # 创建数据库
                conn.execute(
                    text(
                        f"CREATE DATABASE {mysql_config['database']} "
                        f"CHARACTER SET {mysql_config.get('charset', 'utf8mb4')} "
                        f"COLLATE utf8mb4_unicode_ci"
                    )
                )
                conn.commit()
                logger.info(f"✓ 数据库 '{mysql_config['database']}' 创建成功")

        engine.dispose()

    except Exception as e:
        logger.error(f"创建数据库失败: {e}")
        raise


def drop_all_tables(confirm_required=True):
    """
    删除所有表（谨慎使用！）
    仅用于开发环境重置数据库

    Args:
        confirm_required: 是否需要用户确认，默认 True

    Returns:
        bool: 是否成功
    """
    try:
        from settings import MYSQL_CONFIG
        # 直接导入同目录的 models 模块
        import models
        Base = models.Base

        if confirm_required:
            logger.warning("⚠ 警告：即将删除所有表！")
            confirm = input("确认删除所有表？(yes/no): ")
            if confirm.lower() != 'yes':
                logger.info("操作已取消")
                return False

        # 构建连接字符串
        connection_string = _build_connection_string(MYSQL_CONFIG)
        engine = create_engine(connection_string, pool_pre_ping=True)

        # 删除所有表
        Base.metadata.drop_all(engine)

        logger.info("✓ 所有表已删除")
        engine.dispose()

        return True

    except Exception as e:
        logger.error(f"删除表失败: {e}")
        return False


if __name__ == "__main__":
    import argparse

    # 配置命令行日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='数据库初始化工具')
    parser.add_argument(
        '--drop',
        action='store_true',
        help='删除所有表（谨慎使用！）'
    )

    args = parser.parse_args()

    if args.drop:
        # 删除表
        success = drop_all_tables()
        if success:
            logger.info("如需重新创建表，请运行: python -m mysql.init_db")
    else:
        # 初始化数据库
        success = init_database(verbose=True)

        if success:
            logger.info("\n现在可以运行爬虫了！")
            sys.exit(0)
        else:
            sys.exit(1)
