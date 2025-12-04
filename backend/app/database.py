"""
数据库连接和会话管理
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import OperationalError, ProgrammingError
from typing import Generator
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# 创建数据库引擎
engine = create_engine(
    settings.database_url,
    echo=settings.db_echo,
    pool_pre_ping=True,  # 连接池预检查
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,  # 1小时回收连接
)

# 创建会话工厂
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    获取数据库会话
    用于依赖注入
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_and_create_database():
    """
    检查数据库是否存在，不存在则创建
    """
    # 从DATABASE_URL中提取数据库名
    db_name = settings.database_url.split('/')[-1].split('?')[0]

    # 创建不包含数据库名的连接URL（连接到MySQL服务器）
    base_url = settings.database_url.rsplit('/', 1)[0]

    try:
        # 连接到MySQL服务器（不指定数据库）
        temp_engine = create_engine(base_url)

        with temp_engine.connect() as conn:
            # 检查数据库是否存在
            result = conn.execute(text(f"SHOW DATABASES LIKE '{db_name}'"))
            exists = result.fetchone() is not None

            if not exists:
                logger.info(f"数据库 '{db_name}' 不存在，正在创建...")
                conn.execute(text(f"CREATE DATABASE {db_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))
                conn.commit()
                logger.info(f"数据库 '{db_name}' 创建成功")
            else:
                logger.info(f"数据库 '{db_name}' 已存在")

        temp_engine.dispose()
        return True

    except OperationalError as e:
        logger.error(f"无法连接到MySQL服务器: {e}")
        logger.error("请检查MySQL服务是否启动，以及用户名密码是否正确")
        return False
    except Exception as e:
        logger.error(f"创建数据库失败: {e}")
        return False


def init_db():
    """
    初始化数据库表结构

    - 检查并创建数据库
    - 创建所有表
    """
    logger.info("开始初始化数据库...")

    # 1. 检查并创建数据库
    if not check_and_create_database():
        raise Exception("数据库创建失败，请检查MySQL连接")

    # 2. 导入所有模型（确保模型已注册到Base.metadata）
    from app.core.models import task, result, extraction  # noqa

    # 3. 创建所有表
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("数据库表初始化完成")

        # 打印已创建的表
        table_names = Base.metadata.tables.keys()
        logger.info(f"已创建 {len(table_names)} 个表: {', '.join(table_names)}")

    except Exception as e:
        logger.error(f"创建数据库表失败: {e}")
        raise
