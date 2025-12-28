"""简洁的 ReAct Agent 实现（兼容第三方 OpenAI API）"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List

import requests
import yaml
from bs4 import BeautifulSoup
from langchain.tools import Tool
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from zoneinfo import ZoneInfo

from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_CONFIG_PATH = Path("config/settings.yaml")


@dataclass
class AgentStep:
    """推理步骤"""
    log: str
    action: str | None = None
    action_input: Any | None = None
    observation: str | None = None


@dataclass
class AgentRun:
    """执行结果"""
    output: str
    steps: List[AgentStep]


class ReactAgent:
    """简洁的 ReAct Agent（使用生成器模式逐步输出）"""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        system_prompt: str,
        max_iterations: int = 15,
        min_tool_calls: int = 0,
    ):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations
        self.min_tool_calls = min_tool_calls
        self.tool_desc = "\n".join([f"- {name}: {t.description}" for name, t in self.tools.items()])

    def run(self, task: str) -> Generator[dict, None, AgentRun]:
        """执行任务，逐步yield每个步骤的状态"""
        steps = []
        prompt_history = []

        for i in range(self.max_iterations):
            # 构建 prompt
            prompt = self._build_prompt(task, prompt_history)

            # 输出思考状态
            yield {"type": "iteration", "step": i + 1, "max": self.max_iterations}

            # 调用 LLM
            response = self.llm.invoke(prompt)
            text = response.content

            # 输出原始推理内容
            yield {"type": "thinking", "content": text}

            # 先尝试解析并执行 Action（即使文本中有 Final Answer 也先记录工具调用）
            action_result = self._parse_and_execute(text)
            if action_result:
                tool_name, tool_input, observation = action_result
                step = AgentStep(log=text, action=tool_name, action_input=tool_input, observation=observation)
                steps.append(step)
                prompt_history.append(f"{text}\nObservation: {observation}")

                # 输出工具调用
                yield {
                    "type": "action",
                    "tool": tool_name,
                    "input": tool_input,
                }
                # 输出观察结果
                yield {
                    "type": "observation",
                    "content": observation,
                }
            else:
                # 无法解析 action，提示 LLM 重新思考
                error_msg = "Invalid action format. Please use: Action: <tool_name>\\nAction Input: <input>"
                prompt_history.append(f"{text}\nObservation: [{error_msg}]")
                yield {"type": "error", "content": error_msg}

            # 检查是否要结束（在记录完工具调用之后）
            if "Final Answer:" in text:
                # 验证 min_tool_calls
                tool_call_count = len([s for s in steps if s.action])
                if tool_call_count < self.min_tool_calls:
                    warning = (
                        f"需要更多数据支撑。你目前只调用了 {tool_call_count} 次工具，"
                        f"至少需要 {self.min_tool_calls} 次。请继续使用工具收集信息。"
                    )
                    prompt_history.append(f"{text}\nObservation: [{warning}]")
                    yield {"type": "warning", "content": warning}
                    continue

                # 可以输出答案了
                answer = text.split("Final Answer:")[-1].strip()
                yield {"type": "final_answer", "content": answer}
                return AgentRun(output=answer, steps=steps)

        # 达到最大迭代次数
        final_msg = "达到最大迭代次数"
        yield {"type": "max_iterations", "content": final_msg}
        return AgentRun(output=final_msg, steps=steps)

    def _build_prompt(self, task: str, history: List[str]) -> str:
        """构建 ReAct prompt（使用配置的 system_prompt）"""
        # ReAct 框架（不硬编码，从配置读取角色定义）
        base = f"""{self.system_prompt}

Answer the question using this format:

Thought: [reason step by step]
Action: [tool name, must be one of: {', '.join(self.tools.keys())}]
Action Input: [tool input]
Observation: [you will see the result here]
... (repeat Thought/Action/Input/Observation as needed)
Thought: I now know the final answer
Final Answer: [final answer]

Available tools:
{self.tool_desc}

Question: {task}
"""
        if history:
            base += "\n\n" + "\n".join(history) + "\n\nThought:"
        return base

    def _parse_and_execute(self, text: str):
        """解析并执行工具调用"""
        try:
            # 正则解析 Action 和 Action Input
            action_match = re.search(r"Action:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
            input_match = re.search(r"Action Input:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)

            if not action_match or not input_match:
                return None

            tool_name = action_match.group(1).strip()
            tool_input = input_match.group(1).strip()

            # 执行工具
            if tool_name in self.tools:
                try:
                    observation = str(self.tools[tool_name].invoke(tool_input))
                except Exception as e:
                    observation = f"Tool execution error: {str(e)}"
            else:
                observation = f"Error: Tool '{tool_name}' not found. Available: {list(self.tools.keys())}"

            return (tool_name, tool_input, observation)
        except Exception as e:
            return (None, None, f"Parsing error: {e}")


# ============== 配置和工具构建 ==============

def load_settings(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """加载配置文件"""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_chat_model(model_cfg: dict) -> BaseChatModel:
    """构建 ChatOpenAI 模型"""
    api_key = model_cfg.get("api_key") or os.getenv("OPENAI_API_KEY")
    model_name = model_cfg.get("model_name")

    if not api_key or not model_name:
        raise ValueError("需要配置 api_key 和 model_name")

    kwargs = {
        "model": model_name,
        "api_key": api_key,
        "temperature": model_cfg.get("temperature", 0.2),
    }

    # 可选参数
    if base_url := model_cfg.get("base_url"):
        kwargs["base_url"] = base_url
    if max_tokens := model_cfg.get("max_tokens"):
        kwargs["max_tokens"] = max_tokens
    if timeout := model_cfg.get("request_timeout"):
        kwargs["timeout"] = timeout

    return ChatOpenAI(**kwargs)


def build_time_tool(default_tz: str = "Asia/Shanghai") -> Tool:
    """时间查询工具"""
    def get_time(timezone_name: str = "") -> str:
        tz_name = timezone_name.strip() or default_tz
        try:
            tz = ZoneInfo(tz_name)
            return f"{tz_name}: {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')}"
        except Exception as e:
            return f"Invalid timezone '{tz_name}': {e}"

    return Tool(
        name="time_keeper",
        description="查询当前时间或指定时区的时间。输入时区名称（如 Asia/Shanghai）或留空使用默认时区",
        func=get_time,
    )


def build_web_scraper_tool(web_cfg: dict = None) -> Tool:
    """网页抓取工具"""
    web_cfg = web_cfg or {}
    user_agent = web_cfg.get("user_agent", "Mozilla/5.0")
    timeout = web_cfg.get("timeout", 15)
    max_chars = web_cfg.get("max_chars", 4000)

    def scrape(url: str) -> str:
        try:
            headers = {"User-Agent": user_agent}
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            text = " ".join(soup.stripped_strings)
            result = text[:max_chars]
            if len(text) > max_chars:
                result += "..."
            return f"URL: {url}\n内容: {result}"
        except Exception as e:
            return f"Error fetching {url}: {str(e)}"

    return Tool(
        name="web_snapshot",
        description="抓取网页内容并提取文本。输入完整的 URL",
        func=scrape,
    )


def build_mcp_tools(mcp_servers: List[dict]) -> List[BaseTool]:
    """构建 MCP 工具（统一异步加载逻辑）"""

    # 构建服务器配置
    server_configs = {}
    for server in mcp_servers:
        name = server.get("name")
        if not name:
            logger.warning(f"跳过无名称的 MCP 服务器配置: {server}")
            continue

        transport = server.get("transport", "").lower()

        if transport in {"streamable_http", "streamable-http", "http"}:
            server_configs[name] = {
                "transport": "streamable_http",
                "url": server.get("url"),
            }
            if headers := server.get("headers"):
                server_configs[name]["headers"] = headers
        elif transport == "stdio":
            server_configs[name] = {
                "transport": "stdio",
                "command": server.get("command"),
                "args": server.get("args", []),
            }
        else:
            logger.warning(f"不支持的传输类型 '{transport}' for MCP 服务器 '{name}'")

    if not server_configs:
        logger.info("没有有效的 MCP 服务器配置")
        return []

    # 异步加载工具
    async def load_async():
        try:
            logger.info(f"连接 {len(server_configs)} 个 MCP 服务器...")
            client = MultiServerMCPClient(server_configs)
            tools = await client.get_tools()
            logger.info(f"加载 {len(tools)} 个 MCP 工具: {[t.name for t in tools]}")
            return tools
        except Exception as e:
            logger.error(f"MCP 工具加载失败: {e}")
            return []

    return asyncio.run(load_async())


def build_all_tools(settings: dict) -> List[BaseTool]:
    """构建所有工具"""
    tools = []

    # 内置工具
    tools.append(build_time_tool())
    tools.append(build_web_scraper_tool(settings.get("web_scraper")))

    # MCP 工具
    if mcp_servers := settings.get("mcp_servers"):
        tools.extend(build_mcp_tools(mcp_servers))

    return tools


def build_competitor_react_agent(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    max_iterations: int | None = None,
) -> ReactAgent:
    """构建 ReAct Agent"""
    settings = load_settings(Path(config_path))

    # 构建 LLM
    llm = build_chat_model(settings.get("model", {}))

    # 构建工具
    tools = build_all_tools(settings)
    if not tools:
        raise ValueError("至少需要一个工具")
    logger.info(f"加载 {len(tools)} 个工具: {[t.name for t in tools]}")

    # 读取 agent 配置
    agent_cfg = settings.get("agent", {})
    system_prompt = agent_cfg.get("system_prompt", "You are a helpful assistant.")
    iterations = max_iterations or agent_cfg.get("max_iterations", 15)
    min_tool_calls = agent_cfg.get("min_tool_calls", 0)

    return ReactAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        max_iterations=iterations,
        min_tool_calls=min_tool_calls,
    )


__all__ = ["AgentRun", "AgentStep", "ReactAgent", "build_competitor_react_agent"]
