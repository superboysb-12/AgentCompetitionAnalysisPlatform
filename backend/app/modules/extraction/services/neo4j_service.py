"""
Neo4j知识图谱导入服务
支持保留所有三元组属性（source_url, doc_id, evidence等）
"""
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import time

from app.config import settings

logger = logging.getLogger(__name__)


class Neo4jService:
    """Neo4j知识图谱导入服务"""

    def __init__(self):
        """初始化Neo4j连接"""
        self.driver = None
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.password = settings.neo4j_password
        self.database = settings.neo4j_database

    def connect(self):
        """建立Neo4j连接"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=3600
            )
            # 验证连接
            self.driver.verify_connectivity()
            logger.info(f"✓ 成功连接到Neo4j: {self.uri}")
        except AuthError:
            logger.error(f"❌ Neo4j认证失败: 请检查用户名和密码")
            raise
        except ServiceUnavailable:
            logger.error(f"❌ Neo4j服务不可用: {self.uri}")
            raise
        except Exception as e:
            logger.error(f"❌ Neo4j连接失败: {e}")
            raise

    def close(self):
        """关闭Neo4j连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j连接已关闭")

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出"""
        self.close()

    def _sanitize_property_key(self, key: str) -> str:
        """清理属性键名，确保符合Neo4j规范

        Neo4j属性名规则:
        - 不能包含特殊字符（除了下划线）
        - 不能以数字开头
        """
        # 替换特殊字符为下划线
        key = key.replace("-", "_").replace(" ", "_").replace(".", "_")
        # 如果以数字开头，添加前缀
        if key and key[0].isdigit():
            key = f"attr_{key}"
        return key

    def _prepare_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """准备属性字典，保留所有属性

        关键功能：
        1. 保留 source_url, doc_id, evidence 等元数据
        2. 保留未在配置中的动态属性
        3. 清理属性键名，确保符合Neo4j规范
        4. 处理复杂类型（列表、字典等）
        """
        import json
        prepared = {}

        for key, value in properties.items():
            # 清理键名
            clean_key = self._sanitize_property_key(key)

            if value is None:
                continue

            # 处理不同类型的值
            if isinstance(value, (str, int, float, bool)):
                prepared[clean_key] = value
            elif isinstance(value, list):
                # 将列表转换为字符串（Neo4j支持数组，但为了兼容性使用JSON字符串）
                prepared[clean_key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, dict):
                # 将字典转换为JSON字符串
                prepared[clean_key] = json.dumps(value, ensure_ascii=False)
            else:
                # 其他类型转换为字符串
                prepared[clean_key] = str(value)

        return prepared

    def import_triplet(self, triplet: Dict[str, Any]) -> bool:
        """导入单个三元组到Neo4j

        Args:
            triplet: 三元组字典，包含：
                - subject: 主实体名称
                - subject_type: 主实体类型
                - relation: 关系类型
                - object: 客实体名称
                - object_type: 客实体类型
                - source_url: 来源URL
                - doc_id: 文档ID
                - evidence: 证据文本
                - evidence_spans: 证据位置列表
                - confidence: 置信度
                - 以及其他所有未定义的属性

        Returns:
            是否导入成功
        """
        if not self.driver:
            raise RuntimeError("Neo4j连接未建立，请先调用connect()")

        try:
            with self.driver.session(database=self.database) as session:
                # 准备主实体属性（保留所有属性）
                subject_props = self._prepare_properties({
                    'name': triplet.get('subject', ''),
                    'type': triplet.get('subject_type', ''),
                    'source_url': triplet.get('source_url', ''),
                    'doc_id': triplet.get('doc_id', ''),
                })

                # 准备客实体属性（保留所有属性）
                object_props = self._prepare_properties({
                    'name': triplet.get('object', ''),
                    'type': triplet.get('object_type', ''),
                    'source_url': triplet.get('source_url', ''),
                    'doc_id': triplet.get('doc_id', ''),
                })

                # 准备关系属性（保留所有元数据）
                relation_props = self._prepare_properties({
                    'confidence': triplet.get('confidence', 1.0),
                    'evidence': triplet.get('evidence', ''),
                    'source_url': triplet.get('source_url', ''),
                    'doc_id': triplet.get('doc_id', ''),
                    # 保留evidence_spans（转换为JSON字符串）
                    'evidence_spans': triplet.get('evidence_spans', []),
                })

                # 准备关系类型（确保符合Neo4j规范）
                relation_type = self._sanitize_property_key(triplet.get('relation', 'RELATED_TO'))

                # 构建Cypher查询
                # 使用MERGE确保不重复创建节点，SET更新属性
                # CREATE创建关系（允许重复关系，保留不同来源的证据）
                query = f"""
                MERGE (s {{name: $subject_name, type: $subject_type}})
                SET s += $subject_props
                MERGE (o {{name: $object_name, type: $object_type}})
                SET o += $object_props
                CREATE (s)-[r:{relation_type}]->(o)
                SET r = $relation_props
                RETURN s, r, o
                """

                # 执行查询
                result = session.run(
                    query,
                    subject_name=triplet.get('subject', ''),
                    subject_type=triplet.get('subject_type', ''),
                    subject_props=subject_props,
                    object_name=triplet.get('object', ''),
                    object_type=triplet.get('object_type', ''),
                    object_props=object_props,
                    relation_props=relation_props
                )

                # 验证执行结果
                record = result.single()
                if record:
                    logger.debug(f"✓ 导入三元组: ({triplet['subject']})-[{triplet['relation']}]->({triplet['object']})")
                    return True
                else:
                    logger.warning(f"⚠ 三元组导入结果为空: ({triplet['subject']})-[{triplet['relation']}]->({triplet['object']})")
                    return False

        except Exception as e:
            logger.error(f"❌ 导入三元组失败: {e}")
            logger.error(f"三元组内容: {triplet}")
            raise

    def import_triplets_batch(self, triplets: List[Dict[str, Any]]) -> tuple[int, int]:
        """批量导入三元组

        Args:
            triplets: 三元组列表

        Returns:
            (成功数量, 失败数量)
        """
        if not triplets:
            logger.warning("三元组列表为空，跳过导入")
            return 0, 0

        success_count = 0
        failed_count = 0

        logger.info(f"开始批量导入 {len(triplets)} 个三元组...")

        for i, triplet in enumerate(triplets):
            try:
                if self.import_triplet(triplet):
                    success_count += 1
                else:
                    failed_count += 1

                # 每100个记录输出进度
                if (i + 1) % 100 == 0:
                    logger.info(f"进度: {i + 1}/{len(triplets)} (成功: {success_count}, 失败: {failed_count})")

            except Exception as e:
                failed_count += 1
                logger.error(f"导入第 {i+1} 个三元组失败: {e}")

        logger.info(f"批量导入完成: 成功 {success_count}, 失败 {failed_count}")
        return success_count, failed_count

    def import_triplets_with_retry(
        self,
        triplets: List[Dict[str, Any]],
        max_retries: int = None
    ) -> tuple[int, int, Optional[str]]:
        """带重试机制的批量导入

        Args:
            triplets: 三元组列表
            max_retries: 最大重试次数（默认使用配置）

        Returns:
            (成功数量, 失败数量, 错误信息)
        """
        if max_retries is None:
            max_retries = settings.neo4j_max_retries

        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"尝试导入 (第 {attempt + 1}/{max_retries} 次)...")

                # 如果连接断开，重新连接
                if not self.driver:
                    self.connect()

                success_count, failed_count = self.import_triplets_batch(triplets)

                if failed_count == 0:
                    logger.info(f"✓ 全部导入成功")
                    return success_count, failed_count, None
                else:
                    logger.warning(f"部分导入失败: 成功 {success_count}, 失败 {failed_count}")
                    return success_count, failed_count, f"部分导入失败: {failed_count}个"

            except ServiceUnavailable as e:
                last_error = f"Neo4j服务不可用: {e}"
                logger.error(f"❌ {last_error}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("已达到最大重试次数")
                    return 0, len(triplets), last_error

            except Exception as e:
                last_error = f"导入失败: {e}"
                logger.error(f"❌ {last_error}")

                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logger.error("已达到最大重试次数")
                    return 0, len(triplets), last_error

        return 0, len(triplets), last_error

    def test_connection(self) -> bool:
        """测试Neo4j连接

        Returns:
            是否连接成功
        """
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                if record and record["test"] == 1:
                    logger.info("✓ Neo4j连接测试成功")
                    return True
                else:
                    logger.error("❌ Neo4j连接测试失败：返回结果异常")
                    return False
        except Exception as e:
            logger.error(f"❌ Neo4j连接测试失败: {e}")
            return False

    def clear_database(self):
        """清空数据库（谨慎使用！）"""
        logger.warning("⚠️ 正在清空Neo4j数据库...")

        try:
            with self.driver.session(database=self.database) as session:
                # 删除所有节点和关系
                session.run("MATCH (n) DETACH DELETE n")
                logger.info("✓ Neo4j数据库已清空")
        except Exception as e:
            logger.error(f"❌ 清空数据库失败: {e}")
            raise


def create_neo4j_service() -> Neo4jService:
    """创建Neo4j服务实例（工厂函数）"""
    return Neo4jService()
