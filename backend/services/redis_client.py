"""
Redis 客户端模块
提供 Pub/Sub 发布订阅功能和基础 Redis 操作
"""

import json
import redis
from typing import Optional, Dict, Any, Callable
import logging
from settings import REDIS_CONFIG

logger = logging.getLogger(__name__)


class RedisClient:
    """
    Redis 客户端
    支持发布订阅和基础键值操作
    """

    def __init__(self):
        """初始化 Redis 连接池"""
        self.pool = redis.ConnectionPool(
            host=REDIS_CONFIG['host'],
            port=REDIS_CONFIG['port'],
            db=REDIS_CONFIG['db'],
            password=REDIS_CONFIG['password'],
            decode_responses=True,  # 自动解码字符串
            max_connections=10
        )
        self.client = redis.Redis(connection_pool=self.pool)
        logger.info(f"Redis 连接池已创建: {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")

    def publish(self, channel: str, message: Dict[str, Any]) -> int:
        """
        发布消息到指定频道

        Args:
            channel: 频道名称
            message: 消息字典（将被序列化为 JSON）

        Returns:
            int: 接收消息的订阅者数量
        """
        try:
            json_message = json.dumps(message)
            count = self.client.publish(channel, json_message)
            logger.info(f"已发布消息到频道 {channel}, 订阅者数量: {count}")
            return count
        except Exception as e:
            logger.error(f"发布消息失败: {e}")
            raise

    def subscribe(self, channel: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        订阅频道并处理消息（阻塞式）

        Args:
            channel: 频道名称
            callback: 消息处理回调函数，接收解析后的消息字典
        """
        pubsub = self.client.pubsub()
        pubsub.subscribe(channel)
        logger.info(f"已订阅频道: {channel}")

        try:
            for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        data = json.loads(message['data'])
                        logger.debug(f"收到消息: {data}")
                        callback(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"消息解析失败: {e}, 原始数据: {message['data']}")
                    except Exception as e:
                        logger.error(f"消息处理失败: {e}")
        except KeyboardInterrupt:
            logger.info(f"停止订阅频道: {channel}")
        finally:
            pubsub.unsubscribe(channel)
            pubsub.close()

    # ===== 基础键值操作 =====

    def hset_dict(self, key: str, data: Dict[str, Any]) -> None:
        """
        设置 Hash 字典
        兼容旧版本 Redis（< 4.0）
        """
        if not data:
            return

        # 方式1：使用 mapping 参数（Redis 4.0+）
        # 如果失败，回退到逐个设置
        try:
            self.client.hset(key, mapping=data)
        except redis.exceptions.ResponseError:
            # 回退到逐个设置字段
            for field, value in data.items():
                self.client.hset(key, field, value)

    def hget_all(self, key: str) -> Dict[str, str]:
        """获取 Hash 所有字段"""
        return self.client.hgetall(key)

    def hget(self, key: str, field: str) -> Optional[str]:
        """获取 Hash 单个字段"""
        return self.client.hget(key, field)

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return self.client.exists(key) > 0

    def delete(self, key: str) -> None:
        """删除键"""
        self.client.delete(key)

    def keys(self, pattern: str) -> list:
        """获取匹配模式的所有键"""
        return self.client.keys(pattern)

    def close(self) -> None:
        """关闭连接池"""
        self.client.close()
        self.pool.disconnect()
        logger.info("Redis 连接池已关闭")
