"""
自定义OpenAI兼容客户端
支持在URL中传递查询参数（如user_key）的API接口
模仿OpenAI SDK的调用接口，方便无缝集成
"""

import requests
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """消息对象"""
    role: str
    content: str


@dataclass
class Choice:
    """响应选项"""
    index: int
    message: Message
    finish_reason: Optional[str] = None


@dataclass
class Usage:
    """Token使用统计"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class ChatCompletion:
    """聊天完成响应对象，模仿OpenAI SDK的返回格式"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    error: Optional[Dict[str, Any]] = None


class ChatCompletions:
    """聊天完成API，模仿OpenAI SDK的chat.completions接口"""

    def __init__(self, client: 'CustomOpenAIClient'):
        self.client = client

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> ChatCompletion:
        """
        创建聊天完成请求

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            ChatCompletion对象
        """
        return self.client._call_api(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )


class Chat:
    """聊天API，模仿OpenAI SDK的chat接口"""

    def __init__(self, client: 'CustomOpenAIClient'):
        self.completions = ChatCompletions(client)


class CustomOpenAIClient:
    """
    自定义OpenAI兼容客户端
    支持在URL中传递查询参数的API接口

    用法:
        client = CustomOpenAIClient(
            api_key="your-api-key",
            base_url="https://api.example.com/v1/chat/completions",
            url_params={"user_key": "xxxxx"},
            timeout=60
        )

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=2000
        )

        print(response.choices[0].message.content)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        url_params: Optional[Dict[str, str]] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: int = 1
    ):
        """
        初始化自定义客户端

        Args:
            api_key: API密钥
            base_url: API基础URL（完整的endpoint URL）
            url_params: URL查询参数字典，如 {"user_key": "xxxxx"}
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.url_params = url_params or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 创建chat接口
        self.chat = Chat(self)

    def _build_url(self) -> str:
        """构建完整的API URL，包含查询参数"""
        if not self.url_params:
            return self.base_url

        # 构建查询字符串
        params_str = '&'.join([f"{key}={value}" for key, value in self.url_params.items()])

        # 如果base_url已经包含查询参数，使用&连接，否则使用?
        separator = '&' if '?' in self.base_url else '?'
        return f"{self.base_url}{separator}{params_str}"

    def _call_api(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> ChatCompletion:
        """
        调用API

        Args:
            model: 模型名称
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            **kwargs: 其他参数

        Returns:
            ChatCompletion对象
        """
        url = self._build_url()

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        # 重试逻辑
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )

                # 检查HTTP状态码
                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.text}"

                    # 如果是客户端错误(4xx)，不重试
                    if 400 <= response.status_code < 500:
                        return ChatCompletion(
                            id="error",
                            object="chat.completion",
                            created=int(time.time()),
                            model=model,
                            choices=[],
                            usage=None,
                            error={
                                "message": error_msg,
                                "type": "client_error",
                                "code": str(response.status_code)
                            }
                        )

                    # 服务器错误(5xx)，可以重试
                    raise Exception(error_msg)

                # 解析响应
                result = response.json()

                # 检查响应中是否有error字段
                if "error" in result:
                    return ChatCompletion(
                        id=result.get("id", "error"),
                        object="chat.completion",
                        created=int(time.time()),
                        model=model,
                        choices=[],
                        usage=None,
                        error=result["error"]
                    )

                # 构建符合OpenAI SDK格式的响应对象
                choices = []
                for choice_data in result.get("choices", []):
                    message_data = choice_data.get("message", {})
                    message = Message(
                        role=message_data.get("role", "assistant"),
                        content=message_data.get("content", "")
                    )
                    choice = Choice(
                        index=choice_data.get("index", 0),
                        message=message,
                        finish_reason=choice_data.get("finish_reason")
                    )
                    choices.append(choice)

                # 解析usage
                usage = None
                if "usage" in result:
                    usage_data = result["usage"]
                    usage = Usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0)
                    )

                return ChatCompletion(
                    id=result.get("id", "chatcmpl-unknown"),
                    object=result.get("object", "chat.completion"),
                    created=result.get("created", int(time.time())),
                    model=result.get("model", model),
                    choices=choices,
                    usage=usage
                )

            except requests.exceptions.Timeout as e:
                last_error = f"请求超时: {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

            except requests.exceptions.RequestException as e:
                last_error = f"请求失败: {e}"
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue

        # 所有重试都失败了
        return ChatCompletion(
            id="error",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[],
            usage=None,
            error={
                "message": last_error or "未知错误",
                "type": "request_error",
                "code": "max_retries_exceeded"
            }
        )


# 示例用法
if __name__ == "__main__":
    # 创建客户端
    client = CustomOpenAIClient(
        api_key="your-api-key",
        base_url="https://inner-apisix.hisense.com/compatible-openai/v1/chat/completions",
        url_params={"user_key": "xxxxxx"},
        timeout=60
    )

    # 调用API
    response = client.chat.completions.create(
        model="gpt-4-1",
        messages=[
            {"role": "user", "content": "你好，请介绍一下自己"}
        ],
        temperature=0.7,
        max_tokens=500
    )

    # 检查是否有错误
    if response.error:
        print(f"错误: {response.error}")
    else:
        # 打印响应
        print(f"回复: {response.choices[0].message.content}")
        if response.usage:
            print(f"Token使用: {response.usage.total_tokens}")
