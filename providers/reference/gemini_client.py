#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Gemini API 调用 Demo
基于 GeminiProvider.ts 的逻辑实现
支持：流式/非流式、文本/多模态、单文件/多文件、思考内容过滤
"""

import json
import base64
import requests
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class LLMOptions:
    """LLM 调用选项（对应 TypeScript 的 LLMOptions）"""
    api_url: str = ""
    api_key: str = ""
    model: str = "gemini-2.0-pro"
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 8192
    stream: bool = True
    request_timeout_ms: int = 300000


# ============================================================================
# 核心调用类
# ============================================================================

class GeminiClient:
    """
    Google Gemini API 客户端
    对应 TypeScript 的 GeminiProvider
    """

    SYSTEM_PROMPT = "你是一个专业的学术助手，擅长分析和总结学术论文。"

    def __init__(self, options: LLMOptions):
        self.options = options
        self._validate_config()

    def _validate_config(self):
        """验证配置"""
        if not self.options.api_url:
            raise ValueError("Gemini API URL 未配置")
        if not self.options.api_key:
            raise ValueError("Gemini API Key 未配置")

    def _build_headers(self) -> Dict[str, str]:
        """构建请求头"""
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.options.api_key
        }

    def _build_endpoint(self, stream: bool = True) -> str:
        """构建 API 端点"""
        base_url = self.options.api_url.rstrip("/")
        model = self.options.model

        if stream:
            # 流式端点：streamGenerateContent?alt=sse
            return f"{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse"
        else:
            # 非流式端点：generateContent
            return f"{base_url}/v1beta/models/{model}:generateContent"

    def _build_generation_config(self) -> Dict[str, Any]:
        """构建生成配置"""
        config = {}

        if self.options.temperature is not None:
            config["temperature"] = self.options.temperature
        if self.options.top_p is not None:
            config["topP"] = self.options.top_p
        if self.options.max_tokens is not None:
            config["maxOutputTokens"] = self.options.max_tokens

        return config

    def _build_payload(
            self,
            contents: List[Dict[str, Any]],
            stream: bool = True
    ) -> Dict[str, Any]:
        """构建请求体"""
        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": self.SYSTEM_PROMPT}]
            },
            "generationConfig": self._build_generation_config()
        }
        return payload

    def _build_text_content(self, text: str) -> Dict[str, Any]:
        """构建文本内容"""
        return {"text": text}

    def _build_pdf_inline_data(self, base64_content: str) -> Dict[str, Any]:
        """构建 PDF 内联数据（多模态）"""
        return {
            "inline_data": {
                "mime_type": "application/pdf",
                "data": base64_content
            }
        }

    def _build_single_file_contents(
            self,
            prompt: str,
            content: str,
            is_base64: bool = False
    ) -> List[Dict[str, Any]]:
        """构建单文件消息内容"""
        if is_base64:
            # 多模态格式
            return [{
                "role": "user",
                "parts": [
                    {"text": prompt or ""},
                    self._build_pdf_inline_data(content)
                ]
            }]
        else:
            # 文本格式
            user_text = f"{prompt}\n\n{content}" if prompt else content
            return [{
                "role": "user",
                "parts": [{"text": user_text}]
            }]

    def _build_multi_file_contents(
            self,
            pdf_files: List[Dict[str, Any]],
            prompt: str
    ) -> List[Dict[str, Any]]:
        """构建多文件消息内容"""
        parts = [{"text": prompt or ""}]

        for pdf_file in pdf_files:
            base64_content = pdf_file.get("base64Content", "")
            if base64_content:
                parts.append(self._build_pdf_inline_data(base64_content))
                print(f"[INFO] 添加 PDF: {pdf_file.get('displayName', 'unknown')}")

        if len(parts) <= 1:
            raise ValueError("没有成功处理任何 PDF 文件")

        return [{
            "role": "user",
            "parts": parts
        }]

    def _build_conversation_contents(
            self,
            pdf_content: str,
            is_base64: bool,
            conversation: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """构建对话消息内容"""
        contents = []

        if conversation and len(conversation) > 0:
            first_msg = conversation[0]

            # 第一条用户消息附带 PDF 内容
            if is_base64:
                contents.append({
                    "role": "user",
                    "parts": [
                        {"text": first_msg["content"]},
                        self._build_pdf_inline_data(pdf_content)
                    ]
                })
            else:
                user_text = f"{first_msg['content']}\n\n{pdf_content}"
                contents.append({
                    "role": "user",
                    "parts": [{"text": user_text}]
                })

            # 处理后续对话
            if len(conversation) > 1:
                # 第二条是 assistant 回复
                contents.append({
                    "role": "model",
                    "parts": [{"text": conversation[1]["content"]}]
                })

            # 处理剩余对话
            for i in range(2, len(conversation)):
                msg = conversation[i]
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })

        return contents

    def _extract_gemini_text(self, json_data: Dict[str, Any]) -> str:
        """
        提取 Gemini 响应文本
        对应 TypeScript 的 extractGeminiText 方法
        过滤掉 thought: true 的思考内容
        """
        try:
            candidates = json_data.get("candidates", [])
            if not candidates:
                return ""

            candidate = candidates[0]

            # 定义提取函数（过滤 thought 内容）
            def extract_text_from_parts(parts: List[Dict]) -> str:
                if not parts:
                    return ""
                texts = []
                for part in parts:
                    # 过滤掉思考内容
                    if part.get("thought", False):
                        continue
                    text = part.get("text", "")
                    if text:
                        texts.append(text)
                return "".join(texts)

            # 优先从 delta 提取（流式）
            delta = candidate.get("delta", {})
            if delta:
                delta_parts = delta.get("content", {}).get("parts") or delta.get("parts")
                if delta_parts:
                    return extract_text_from_parts(delta_parts)

            # 从 content 提取（非流式或完整响应）
            content = candidate.get("content", {})
            if content:
                parts = content.get("parts", [])
                if parts:
                    return extract_text_from_parts(parts)

            return ""
        except Exception as e:
            print(f"[WARN] 提取 Gemini 文本失败：{e}")
            return ""

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
        contents = self._build_single_file_contents(prompt, content, is_base64)
        payload = self._build_payload(contents, stream=self.options.stream)

        if self.options.stream and on_progress:
            return self._stream_request(payload, on_progress)
        else:
            return self._non_stream_request(payload)

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
        contents = self._build_conversation_contents(
            pdf_content, is_base64, conversation
        )
        payload = self._build_payload(contents, stream=self.options.stream)

        if self.options.stream and on_progress:
            return self._stream_request(payload, on_progress)
        else:
            return self._non_stream_request(payload)

    def generate_multi_file_summary(
            self,
            pdf_files: List[Dict[str, Any]],
            prompt: str,
            on_progress: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        多文件摘要生成（对应 generateMultiFileSummary）

        Args:
            pdf_files: PDF 文件列表，每项包含 base64Content 和 displayName
            prompt: 提示词
            on_progress: 流式回调函数
        """
        contents = self._build_multi_file_contents(pdf_files, prompt)
        payload = self._build_payload(contents, stream=self.options.stream)

        if self.options.stream and on_progress:
            return self._stream_request(payload, on_progress)
        else:
            return self._non_stream_request(payload)

    def test_connection(self) -> str:
        """
        测试连接（对应 testConnection）
        """
        contents = [{
            "role": "user",
            "parts": [{
                "text": "Hello! Please respond with 'OK' to confirm connection."
            }]
        }]

        payload = {
            "contents": contents,
            "systemInstruction": {
                "parts": [{"text": self.SYSTEM_PROMPT}]
            },
            "generationConfig": {
                "temperature": 0.1,
                "topP": 1.0,
                "maxOutputTokens": 16
            }
        }

        endpoint = self._build_endpoint(stream=False)

        try:
            response = requests.post(
                endpoint,
                headers=self._build_headers(),
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            data = response.json()
            text = self._extract_gemini_text(data)

            return f"✅ 连接成功!\n模型：{self.options.model}\n响应：{text}"
        except requests.exceptions.RequestException as e:
            error_msg = self._parse_error(e)
            raise Exception(f"连接测试失败：{error_msg}")

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _non_stream_request(self, payload: Dict[str, Any]) -> str:
        """非流式请求"""
        endpoint = self._build_endpoint(stream=False)

        try:
            response = requests.post(
                endpoint,
                headers=self._build_headers(),
                json=payload,
                timeout=self.options.request_timeout_ms / 1000
            )
            response.raise_for_status()

            data = response.json()
            return self._extract_gemini_text(data)
        except requests.exceptions.RequestException as e:
            error_msg = self._parse_error(e)
            raise Exception(f"请求失败：{error_msg}")

    def _stream_request(
            self,
            payload: Dict[str, Any],
            on_progress: Callable[[str], None]
    ) -> str:
        """
        流式请求（SSE）

        对应 TypeScript 中的 Zotero.HTTP.request + requestObserver
        """
        endpoint = self._build_endpoint(stream=True)
        chunks = []

        try:
            response = requests.post(
                endpoint,
                headers=self._build_headers(),
                json=payload,
                stream=True,
                timeout=self.options.request_timeout_ms / 1000
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode('utf-8')

                # 解析 SSE 格式：data: {json}
                if not line_str.startswith("data:"):
                    continue

                data_str = line_str[5:].strip()  # 去掉 "data:" 前缀

                if not data_str:
                    continue

                try:
                    json_data = json.loads(data_str)
                    text = self._extract_gemini_text(json_data)

                    if text:
                        chunks.append(text)
                        on_progress(text)
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
        api_url="https://generativelanguage.googleapis.com",
        api_key="your-gemini-api-key",  # 替换为你的 API Key
        model="gemini-2.0-pro",
        temperature=0.7,
        stream=True
    )

    client = GeminiClient(options)

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
        api_url="https://generativelanguage.googleapis.com",
        api_key="your-gemini-api-key",
        model="gemini-2.0-pro",
        stream=True
    )

    client = GeminiClient(options)

    # 读取 PDF 并转为 Base64
    with open("document.pdf", "rb") as f:
        pdf_base64 = base64.b64encode(f.read()).decode('utf-8')

    def on_progress(chunk: str):
        print(chunk, end="", flush=True)

    result = client.generate_summary(
        content=pdf_base64,
        is_base64=True,
        prompt="请分析这个 PDF 文档",
        on_progress=on_progress
    )

    print(f"\n\n结果长度：{len(result)} 字符")


def example_multi_file_summary():
    """示例 3: 多文件摘要"""
    print("=" * 60)
    print("示例 3: 多文件摘要")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://generativelanguage.googleapis.com",
        api_key="your-gemini-api-key",
        model="gemini-2.0-pro",
        stream=True
    )

    client = GeminiClient(options)

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

    def on_progress(chunk: str):
        print(chunk, end="", flush=True)

    result = client.generate_multi_file_summary(
        pdf_files=pdf_files,
        prompt="请综合总结这些 PDF 文件的共同点和差异",
        on_progress=on_progress
    )

    print(f"\n\n结果长度：{len(result)} 字符")


def example_chat():
    """示例 4: 对话聊天"""
    print("=" * 60)
    print("示例 4: 对话聊天")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://generativelanguage.googleapis.com",
        api_key="your-gemini-api-key",
        model="gemini-2.0-pro",
        stream=True
    )

    client = GeminiClient(options)

    paper_content = "这是一篇关于机器学习的论文..."

    conversation = [
        {"role": "user", "content": "请总结这篇论文"},
        {"role": "assistant", "content": "这篇论文的主要贡献是..."},
        {"role": "user", "content": "能详细解释一下方法部分吗？"}
    ]

    def on_progress(chunk: str):
        print(chunk, end="", flush=True)

    result = client.chat(
        pdf_content=paper_content,
        is_base64=False,
        conversation=conversation,
        on_progress=on_progress
    )

    print(f"\n\n结果：{result}")


def example_test_connection():
    """示例 5: 测试连接"""
    print("=" * 60)
    print("示例 5: 测试连接")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://generativelanguage.googleapis.com",
        api_key="your-gemini-api-key",
        model="gemini-2.0-pro",
        stream=False
    )

    client = GeminiClient(options)

    try:
        result = client.test_connection()
        print(result)
    except Exception as e:
        print(f"❌ 连接失败：{e}")


def example_thinking_model():
    """示例 6: 使用思考模型（自动过滤思考内容）"""
    print("=" * 60)
    print("示例 6: 使用思考模型")
    print("=" * 60)

    options = LLMOptions(
        api_url="https://generativelanguage.googleapis.com",
        api_key="your-gemini-api-key",
        model="gemini-2.5-pro",  # 支持思考的模型
        temperature=0.7,
        stream=True
    )

    client = GeminiClient(options)

    def on_progress(chunk: str):
        print(chunk, end="", flush=True)

    # _extract_gemini_text 会自动过滤 thought: true 的内容
    result = client.generate_summary(
        content="请解释量子计算的基本原理",
        on_progress=on_progress
    )

    print(f"\n\n（思考过程已自动过滤，只显示最终答案）")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 运行示例（取消注释以测试）
    # example_test_connection()
    # example_single_text_summary()
    # example_chat()

    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║            Google Gemini API 调用 Demo                     ║
    ╠══════════════════════════════════════════════════════════╣
    ║  功能：                                                   ║
    ║  • 单文件文本摘要                                          ║
    ║  • 单文件 Base64 多模态摘要                                 ║
    ║  • 多文件摘要                                              ║
    ║  • 对话聊天                                                ║
    ║  • 连接测试                                                ║
    ║  • 思考内容自动过滤                                        ║
    ╠══════════════════════════════════════════════════════════╣
    ║  使用方法：                                               ║
    ║  1. 替换 api_key 为你的实际 Gemini API Key                  ║
    ║  2. 取消注释 main 中的示例函数                              ║
    ║  3. 运行：python gemini_demo.py                           ║
    ╠══════════════════════════════════════════════════════════╣
    ║  获取 API Key:                                            ║
    ║  https://aistudio.google.com/apikey                       ║
    ╚══════════════════════════════════════════════════════════╝
    """)