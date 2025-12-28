"""Simple harness that runs the ReAct agent and prints its reasoning trace.

Usage:
    python test_agent.py "写一份针对中国空调行业的竞品分析框架"

The script loads the settings from config/settings.yaml, builds the agent,
executes a single task, and prints each ReAct step (thought, tool call,
observation) followed by the final answer. This makes it easy to verify that
the MCP + 网页抓取 + 时间 工具都被调用。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

from react_agent import AgentRun, build_competitor_react_agent


def _print_step_event(event: dict) -> None:
    """打印单个步骤事件"""
    event_type = event.get("type")

    if event_type == "iteration":
        step_num = event.get("step")
        max_steps = event.get("max")
        print(f"\n{'='*60}")
        print(f"Step {step_num}/{max_steps}")
        print(f"{'='*60}")

    elif event_type == "thinking":
        content = event.get("content", "")
        print(f"\n[THINKING]")
        for line in content.strip().split("\n"):
            print(f"  {line}")

    elif event_type == "action":
        tool = event.get("tool")
        tool_input = event.get("input")
        print(f"\n[ACTION] {tool}")
        print(f"  Input: {tool_input}")

    elif event_type == "observation":
        content = event.get("content", "")
        # 截断过长的观察结果
        if len(content) > 500:
            content = content[:500] + "..."
        print(f"\n[OBSERVATION]")
        for line in content.strip().split("\n"):
            print(f"  {line}")

    elif event_type == "warning":
        content = event.get("content", "")
        print(f"\n[WARNING] {content}")

    elif event_type == "error":
        content = event.get("content", "")
        print(f"\n[ERROR] {content}")

    elif event_type == "final_answer":
        content = event.get("content", "")
        print(f"\n{'='*60}")
        print("[FINAL ANSWER]")
        print(f"{'='*60}")
        print(content)

    elif event_type == "max_iterations":
        content = event.get("content", "")
        print(f"\n[TIMEOUT] {content}")


def _run_once(agent, task: str) -> AgentRun:
    """运行一次任务，打印所有步骤"""
    result = None
    for event in agent.run(task):
        if isinstance(event, AgentRun):
            # 最后返回的是 AgentRun 对象
            result = event
        else:
            # 中间步骤是字典
            _print_step_event(event)

    return result


def _format_chat_task(history: List[Dict[str, str]], new_prompt: str, max_turns: int) -> str:
    """格式化多轮对话的任务"""
    if not history:
        return new_prompt
    trimmed = history[-max(0, max_turns):]
    transcript_lines = []
    for idx, turn in enumerate(trimmed, start=1):
        transcript_lines.append(
            f"[Round {idx}] 用户: {turn['user']}\n[Round {idx}] Agent: {turn['agent']}"
        )
    transcript = "\n".join(transcript_lines)
    return (
        "以下是此前对话的摘要，请在保持上下文一致的前提下继续：\n"
        f"{transcript}\n\n当前用户新任务：{new_prompt}"
    )


def _chat_loop(agent, initial_prompt: str, history_turns: int) -> None:
    """多轮交互模式"""
    print("进入多轮交互模式，输入 exit/quit 可退出。")
    history: List[Dict[str, str]] = []

    def _execute(user_prompt: str) -> None:
        task = _format_chat_task(history, user_prompt, history_turns)
        result = _run_once(agent, task)
        if result:
            history.append({"user": user_prompt, "agent": result.output})

    if initial_prompt:
        _execute(initial_prompt)

    while True:
        try:
            user_prompt = input("\nUser> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出多轮模式。")
            break
        if not user_prompt:
            continue
        if user_prompt.lower() in {"exit", "quit"}:
            print("已退出多轮模式。")
            break
        _execute(user_prompt)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LangChain ReAct agent and show intermediate reasoning"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="请输出一份空调竞品分析报告框架，并说明每个章节需要的数据来源",
        help="任务描述，默认为空调竞品分析示例",
    )
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="ReAct 最大思考步数（覆盖配置文件）",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="进入多轮对话模式，可持续跟进上一轮结果",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=4,
        help="多轮模式下携带的历史轮次数量",
    )
    args = parser.parse_args()

    # 构建 agent
    agent = build_competitor_react_agent(
        config_path=Path(args.config),
        max_iterations=args.max_iterations,
    )

    # 运行
    if args.chat:
        _chat_loop(agent, args.prompt, args.history_turns)
    else:
        _run_once(agent, args.prompt)


if __name__ == "__main__":
    main()
