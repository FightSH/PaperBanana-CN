#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenAI 兼容 API 调用 Demo
基于 OpenAICompatProvider.ts 的逻辑实现
支持：流式/非流式、文本/多模态、单文件/多文件
"""

import json
import requests
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class LLMOptions:
    """LLM 调用选项（对应 TypeScript 的 LLMOptions）"""
    api_url: str = "https://api.openai.com/v1/chat/completions"
    api_key: str = ""
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 4096
    stream: bool = True
    request_timeout_ms: int = 300000


# ============================================================================
# 核心调用类
# ============================================================================

class OpenAICompatClient:
    """
    OpenAI 兼容 API 客户端
    对应 TypeScript 的 OpenAICompatProvider
    """

    SYSTEM_PROMPT = "你是一个专业的学术助手，擅长分析和总结学术论文。"

    def __init__(self, options: LLMOptions):
        self.options = options
        self._validate_config()

    def _validate_config(self):
        """验证配置"""
        if not self.options.api_url:
            raise ValueError("API URL 未配置")
        if not self.options.api_key:
            raise ValueError("API Key 未配置")

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.options.api_key}"
        }

    def _build_payload(
            self,
            messages: List[Dict[str, Any]],
            stream: bool = True
    ) -> Dict[str, Any]:
        """构建请求体"""
        payload = {
            "model": self.options.model,
            "messages": messages,
            "stream": stream
        }

        # 仅在设置时添加可选参数
        if self.options.temperature is not None:
            payload["temperature"] = self.options.temperature
        if self.options.top_p is not None:
            payload["top_p"] = self.options.top_p
        if self.options.max_tokens is not None:
            payload["max_tokens"] = self.options.max_tokens

        return payload

    def _build_messages(
            self,
            prompt: str,
            content: str,
            is_base64: bool = False
    ) -> List[Dict[str, Any]]:
        """
        构建消息数组

        Args:
            prompt: 用户提示词
            content: 内容（文本或 Base64）
            is_base64: 是否为 Base64 格式
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        if is_base64:
            # 多模态格式
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt or "请分析这个文档。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:application/pdf;base64,{content}"
                        }
                    }
                ]
            })
        else:
            # 文本格式
            user_text = f"{prompt}\n\n{content}" if prompt else content
            messages.append({
                "role": "user",
                "content": user_text
            })

        return messages

    def _build_multi_file_messages(
            self,
            pdf_files: List[Dict[str, Any]],
            prompt: str
    ) -> List[Dict[str, Any]]:
        """
        构建多文件消息数组

        Args:
            pdf_files: PDF 文件列表，每项包含 base64Content 和 displayName
            prompt: 用户提示词
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        file_parts = []
        for pdf_file in pdf_files:
            base64_content = pdf_file.get("base64Content", "")
            if base64_content:
                file_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:application/pdf;base64,{base64_content}"
                    }
                })
                print(f"[INFO] 添加 PDF: {pdf_file.get('displayName', 'unknown')}")

        if not file_parts:
            raise ValueError("没有成功处理任何 PDF 文件")

        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                *file_parts
            ]
        })

        return messages

    # ========================================================================
    # 公开方法
    # ========================================================================

    def generate_summary(
            self,
            content: str,
            is_base64: bool = False,
            prompt: Optional[str] = None,
            on_progress: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        生成摘要（对应 generateSummary）

        Args:
            content: 内容（文本或 Base64）
            is_base64: 是否为 Base64 格式
            prompt: 提示词
            on_progress: 流式回调函数
        """
        messages = self._build_messages(prompt, content, is_base64)

        if self.options.stream and on_progress:
            return self._stream_request(messages, on_progress)
        else:
            return self._non_stream_request(messages)

    def chat(
            self,
            pdf_content: str,
            is_base64: bool,
            conversation: List[Dict[str, str]],
            on_progress: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        对话聊天（对应 chat）

        Args:
            pdf_content: PDF 内容（文本或 Base64）
            is_base64: 是否为 Base64 格式
            conversation: 对话历史 [{role, content}, ...]
            on_progress: 流式回调函数
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        for i, msg in enumerate(conversation):
            role = msg.get("role", "user")
            if role not in ["system", "user", "assistant"]:
                role = "user"

            # 第一条用户消息附带 PDF 内容
            if role == "user" and i == 0:
                if is_base64:
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": msg["content"]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:application/pdf;base64,{pdf_content}"
                                }
                            }
                        ]
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"{msg['content']}\n\n{pdf_content}"
                    })
            else:
                messages.append({
                    "role": role,
                    "content": msg["content"]
                })

        if self.options.stream and on_progress:
            return self._stream_request(messages, on_progress)
        else:
            return self._non_stream_request(messages)

    def generate_multi_file_summary(
            self,
            pdf_files: List[Dict[str, Any]],
            prompt: str,
            on_progress: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        多文件摘要生成（对应 generateMultiFileSummary）

        Args:
            pdf_files: PDF 文件列表
            prompt: 提示词
            on_progress: 流式回调函数
        """
        messages = self._build_multi_file_messages(pdf_files, prompt)

        if self.options.stream and on_progress:
            return self._stream_request(messages, on_progress)
        else:
            return self._non_stream_request(messages)

    def test_connection(self) -> str:
        """
        测试连接（对应 testConnection）
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Hello! I want to know who are you?."
            }
        ]

        try:
            response = requests.post(
                self.options.api_url,
                headers=self._build_headers(),
                json=self._build_payload(messages, stream=False),
                timeout=self.options.request_timeout_ms / 1000
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            return f"✅ 连接成功!\n模型：{self.options.model}\n响应：{content}"
        except requests.exceptions.RequestException as e:
            error_msg = self._parse_error(e)
            raise Exception(f"连接测试失败：{error_msg}")

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _non_stream_request(self, messages: List[Dict[str, Any]]) -> str:
        """非流式请求"""
        try:
            response = requests.post(
                self.options.api_url,
                headers=self._build_headers(),
                json=self._build_payload(messages, stream=False),
                timeout=self.options.request_timeout_ms / 1000
            )
            response.raise_for_status()

            data = response.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
        except requests.exceptions.RequestException as e:
            error_msg = self._parse_error(e)
            raise Exception(f"请求失败：{error_msg}")

    def _stream_request(
            self,
            messages: List[Dict[str, Any]],
            on_progress: Callable[[str], None]
    ) -> str:
        """
        流式请求（SSE）

        对应 TypeScript 中的 Zotero.HTTP.request + requestObserver
        """
        chunks = []

        try:
            response = requests.post(
                self.options.api_url,
                headers=self._build_headers(),
                json=self._build_payload(messages, stream=True),
                stream=True,
                timeout=self.options.request_timeout_ms / 1000
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8')

                # 解析 SSE 格式
                if not line_str.startswith("data:"):
                    continue

                data_str = line_str[5:].strip()  # 去掉 "data:" 前缀

                if not data_str or data_str == "[DONE]":
                    continue

                try:
                    event = json.loads(data_str)
                    delta = (
                        event.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )

                    if delta:
                        chunks.append(delta)
                        on_progress(delta)
                except json.JSONDecodeError:
                    continue

            return "".join(chunks)
        except requests.exceptions.RequestException as e:
            if chunks:
                # 已有部分输出，返回已收集的内容
                return "".join(chunks)
            error_msg = self._parse_error(e)
            raise Exception(f"流式请求失败：{error_msg}")

    def _parse_error(self, error: requests.exceptions.RequestException) -> str:
        """解析错误信息"""
        if hasattr(error, 'response') and error.response is not None:
            try:
                error_data = error.response.json()
                err = error_data.get("error", error_data)
                code = err.get("code", f"HTTP {error.response.status_code}")
                msg = err.get("message", str(error))
                return f"{code}: {msg}"
            except:
                pass
        return str(error)


# ============================================================================
# 使用示例
# ============================================================================

def example_single_text_summary():
    """示例 1: 单文件文本摘要"""
    print("=" * 60)
    print("示例 1: 单文件文本摘要")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key="sk-your-api-key-here",  # 替换为你的 API Key
        model="gpt-3.5-turbo",
        temperature=0.7,
        stream=True
    )

    client = OpenAICompatClient(options)

    # 模拟论文内容
    paper_content = """
    本文研究了深度学习在自然语言处理中的应用...
    （此处为论文全文内容）
    """

    def on_progress(chunk: str):
        print(chunk, end="", flush=True)

    result = client.generate_summary(
        content=paper_content,
        is_base64=False,
        prompt="请总结这篇论文的主要贡献和方法",
        on_progress=on_progress
    )

    print(f"\n\n最终结果长度：{len(result)} 字符")


def example_single_base64_summary():
    """示例 2: 单文件 Base64 多模态摘要"""
    print("=" * 60)
    print("示例 2: 单文件 Base64 多模态摘要")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key="sk-your-api-key-here",
        model="gpt-4-vision-preview",
        stream=True
    )

    client = OpenAICompatClient(options)

    # 读取 PDF 并转为 Base64
    import base64
    with open("document.pdf", "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode('utf-8')

    result = client.generate_summary(
        content=pdf_base64,
        is_base64=True,
        prompt="请分析这个 PDF 文档"
    )

    print(f"\n结果：{result[:200]}...")


def example_multi_file_summary():
    """示例 3: 多文件摘要"""
    print("=" * 60)
    print("示例 3: 多文件摘要")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key="sk-your-api-key-here",
        model="gpt-4-vision-preview",
        stream=True
    )

    client = OpenAICompatClient(options)

    import base64

    pdf_files = [
        {
            "filePath": "/path/to/paper1.pdf",
            "displayName": "论文 1.pdf",
            "base64Content": base64.b64encode(open("paper1.pdf", "rb").read()).decode('utf-8')
        },
        {
            "filePath": "/path/to/paper2.pdf",
            "displayName": "论文 2.pdf",
            "base64Content": base64.b64encode(open("paper2.pdf", "rb").read()).decode('utf-8')
        }
    ]

    result = client.generate_multi_file_summary(
        pdf_files=pdf_files,
        prompt="请综合总结这些 PDF 文件的共同点和差异"
    )

    print(f"\n结果：{result[:200]}...")


def example_chat():
    """示例 4: 对话聊天"""
    print("=" * 60)
    print("示例 4: 对话聊天")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key="sk-your-api-key-here",
        model="gpt-3.5-turbo",
        stream=True
    )

    client = OpenAICompatClient(options)

    paper_content = "这是一篇关于机器学习的论文..."

    conversation = [
        {"role": "user", "content": "请总结这篇论文"},
        {"role": "assistant", "content": "这篇论文的主要贡献是..."},
        {"role": "user", "content": "能详细解释一下方法部分吗？"}
    ]

    result = client.chat(
        pdf_content=paper_content,
        is_base64=False,
        conversation=conversation
    )

    print(f"\n结果：{result}")


def example_test_connection():
    """示例 5: 测试连接"""
    print("=" * 60)
    print("示例 5: 测试连接")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://api.openai.com/v1/chat/completions",
        api_key="sk-your-api-key-here",
        model="gpt-3.5-turbo",
        stream=False
    )

    client = OpenAICompatClient(options)

    try:
        result = client.test_connection()
        print(result)
    except Exception as e:
        print(f"❌ 连接失败：{e}")


def example_third_party_api():
    """示例 6: 第三方兼容 API（如 SiliconFlow、DeepSeek 等）"""
    print("=" * 60)
    print("示例 6: 第三方兼容 API")
    print("=" * 60)

    # SiliconFlow 示例
    options = LLMOptions(
        api_url="https://api.siliconflow.cn/v1/chat/completions",
        api_key="sk-siliconflow-key",
        model="deepseek-ai/DeepSeek-V3",
        temperature=0.7,
        stream=True
    )

    client = OpenAICompatClient(options)

    result = client.generate_summary(
        content="请解释什么是Transformer架构",
        is_base64=False,
        prompt="请详细解释"
    )

    print(f"\n结果：{result[:200]}...")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 运行示例（取消注释以测试）
    # example_test_connection()
    # example_single_text_summary()
    # example_chat()
    # example_third_party_api()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           OpenAI 兼容 API 调用 Demo                        ║
    ╠══════════════════════════════════════════════════════════╣
    ║  功能：                                                   ║
    ║  • 单文件文本摘要                                          ║
    ║  • 单文件 Base64 多模态摘要                                 ║
    ║  • 多文件摘要                                              ║
    ║  • 对话聊天                                                ║
    ║  • 连接测试                                                ║
    ║  • 第三方 API 兼容（SiliconFlow、DeepSeek 等）               ║
    ╠══════════════════════════════════════════════════════════╣
    ║  使用方法：                                               ║
    ║  1. 替换 api_key 为你的实际 API Key                         ║
    ║  2. 取消注释 main 中的示例函数                              ║
    ║  3. 运行：python openai_compat_demo.py                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)