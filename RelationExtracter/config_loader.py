"""
配置文件加载器
用于加载和管理YAML配置文件
"""

import yaml
import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置文件加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，默认为当前目录下的config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        self.config_path = Path(config_path)
        self._config_data = None
        self.load_config()

    def load_config(self) -> None:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            logger.info(f"配置文件加载成功: {self.config_path}")
        except FileNotFoundError:
            logger.error(f"配置文件未找到: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"配置文件格式错误: {e}")
            raise
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise

    def reload_config(self) -> None:
        """重新加载配置文件"""
        logger.info("重新加载配置文件...")
        self.load_config()

    def get_schema(self, schema_key: str) -> Dict[str, List[str]]:
        """
        获取指定的schema

        Args:
            schema_key: schema键名

        Returns:
            schema字典
        """
        schemas = self._config_data.get("schemas", {})
        if schema_key not in schemas:
            available_keys = list(schemas.keys())
            raise ValueError(f"Schema '{schema_key}' 不存在。可用的schema: {available_keys}")

        return schemas[schema_key]

    def get_config(self, config_name: str) -> Dict[str, Any]:
        """
        获取指定的配置

        Args:
            config_name: 配置名称

        Returns:
            配置字典
        """
        configs = self._config_data.get("configs", {})
        if config_name not in configs:
            available_configs = list(configs.keys())
            raise ValueError(f"配置 '{config_name}' 不存在。可用的配置: {available_configs}")

        config = configs[config_name].copy()

        # 解析schema
        schema_key = config.get("schema_key")
        if schema_key:
            config["schema"] = self.get_schema(schema_key)

        return config

    def list_schemas(self) -> List[str]:
        """获取所有可用的schema名称"""
        return list(self._config_data.get("schemas", {}).keys())

    def list_configs(self) -> List[str]:
        """获取所有可用的配置名称"""
        return list(self._config_data.get("configs", {}).keys())

    def get_model_config(self, config_name: str) -> Dict[str, Any]:
        """获取模型配置部分"""
        config = self.get_config(config_name)
        return config.get("model", {})

    def get_text_processing_config(self, config_name: str) -> Dict[str, Any]:
        """获取文本处理配置部分"""
        config = self.get_config(config_name)
        return config.get("text_processing", {})

    def add_custom_schema(self, schema_name: str, schema_data: Dict[str, List[str]]) -> None:
        """
        添加自定义schema（运行时添加，不保存到文件）

        Args:
            schema_name: schema名称
            schema_data: schema数据
        """
        if "schemas" not in self._config_data:
            self._config_data["schemas"] = {}

        self._config_data["schemas"][schema_name] = schema_data
        logger.info(f"添加自定义schema: {schema_name}")

    def create_custom_config(
        self,
        config_name: str,
        schema_key: str,
        model_overrides: Optional[Dict[str, Any]] = None,
        text_processing_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建自定义配置

        Args:
            config_name: 基础配置名称
            schema_key: 使用的schema键
            model_overrides: 模型配置覆盖
            text_processing_overrides: 文本处理配置覆盖

        Returns:
            自定义配置字典
        """
        base_config = self.get_config(config_name)

        # 更新schema
        base_config["schema_key"] = schema_key
        base_config["schema"] = self.get_schema(schema_key)

        # 更新模型配置
        if model_overrides:
            base_config["model"].update(model_overrides)

        # 更新文本处理配置
        if text_processing_overrides:
            base_config["text_processing"].update(text_processing_overrides)

        return base_config


# 全局配置加载器实例
_config_loader = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """
    获取全局配置加载器实例

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigLoader实例
    """
    global _config_loader
    if _config_loader is None or config_path is not None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def get_schema(schema_key: str) -> Dict[str, List[str]]:
    """快捷函数：获取schema"""
    return get_config_loader().get_schema(schema_key)


def get_config(config_name: str) -> Dict[str, Any]:
    """快捷函数：获取配置"""
    return get_config_loader().get_config(config_name)


def list_available_schemas() -> List[str]:
    """快捷函数：列出可用schema"""
    return get_config_loader().list_schemas()


def list_available_configs() -> List[str]:
    """快捷函数：列出可用配置"""
    return get_config_loader().list_configs()