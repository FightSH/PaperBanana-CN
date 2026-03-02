"""
MultiProvider — 文本和图像使用不同 API 后端的混合 Provider

支持将文本生成和图像生成分别路由到不同的 API 端点，
每个端点可以独立选择 OpenAI 兼容 或 Gemini 原生 接口风格。

实现 BaseProvider 接口，与 EvolinkProvider 并存，
通过 provider="multi" 切换。
"""

import asyncio
import base64
import json
import re
from typing import List, Dict, Any, Optional

import aiohttp

from .base import BaseProvider


class ClientError(Exception):
    """4xx 客户端错误，不应重试"""
    pass


class MultiProvider(BaseProvider):
    """
    混合 Provider：文本和图像可使用不同的 API 后端。

    每个通道（text / image）独立配置：
      - api_style: "openai" 或 "gemini"
      - api_key: 对应的 API 密钥
      - base_url: API 基础地址
      - image_openai_endpoint: 当 image api_style=openai 时，
        选择 "chat" / "images" / "auto" 路由策略
    """

    def __init__(
        self,
        text_api_style: str,
        text_api_key: str,
        text_base_url: str,
        image_api_style: str,
        image_api_key: str,
        image_base_url: str,
        image_openai_endpoint: str = "auto",
    ):
        self.text_api_style = text_api_style.lower().strip()
        self.text_api_key = text_api_key
        self.text_base_url = text_base_url.rstrip("/")

        self.image_api_style = image_api_style.lower().strip()
        self.image_api_key = image_api_key
        self.image_base_url = image_base_url.rstrip("/")
        endpoint_mode = str(image_openai_endpoint or "auto").lower().strip()
        if endpoint_mode not in ("auto", "chat", "images"):
            print(f"[MultiProvider] 未知 image_openai_endpoint={image_openai_endpoint}，回退为 auto")
            endpoint_mode = "auto"
        self.image_openai_endpoint = endpoint_mode

        self._session: Optional[aiohttp.ClientSession] = None

    # ==================== Session 管理 ====================

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=30)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _sanitize_headers_for_log(self, headers: Dict[str, str]) -> Dict[str, str]:
        """日志脱敏：隐藏敏感头信息。"""
        safe = dict(headers)
        auth = safe.get("Authorization")
        if isinstance(auth, str) and auth:
            safe["Authorization"] = "Bearer ***"
        api_key = safe.get("x-goog-api-key")
        if isinstance(api_key, str) and api_key:
            safe["x-goog-api-key"] = "***"
        return safe

    def _is_sensitive_key(self, key: str) -> bool:
        if not isinstance(key, str):
            return False
        k = key.lower()
        return any(token in k for token in ("key", "token", "secret", "authorization", "password"))

    def _looks_like_base64(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        stripped = value.strip()
        if len(stripped) < 512:
            return False
        if stripped.startswith("data:") and ";base64," in stripped:
            return True
        return bool(re.match(r"^[A-Za-z0-9+/=\s]+$", stripped))

    def _summarize_for_log(self, value: Any, parent_key: str = "", max_str_len: int = 240) -> Any:
        """
        递归裁剪日志内容，避免超长参数（如 base64）刷屏。
        仅用于日志展示，不影响真实请求体。
        """
        if isinstance(value, dict):
            summarized = {}
            for k, v in value.items():
                if self._is_sensitive_key(k):
                    summarized[k] = "***"
                else:
                    summarized[k] = self._summarize_for_log(
                        v, parent_key=str(k), max_str_len=max_str_len
                    )
            return summarized
        if isinstance(value, list):
            return [self._summarize_for_log(v, parent_key=parent_key, max_str_len=max_str_len) for v in value]
        if isinstance(value, str):
            # data URL（常见于多模态图片）仅打印长度与前缀
            if value.startswith("data:") and ";base64," in value:
                prefix, b64 = value.split(";base64,", 1)
                return f"{prefix};base64,<omitted len={len(b64)}>"
            if self._is_sensitive_key(parent_key):
                return "***"
            if self._looks_like_base64(value):
                compact = re.sub(r"\s+", "", value)
                return f"<base64 omitted len={len(compact)}>"
            if len(value) > max_str_len:
                return f"{value[:max_str_len]}...(len={len(value)})"
        return value

    def _dump_response_for_log(self, response: Any) -> str:
        try:
            sanitized = self._summarize_for_log(response, max_str_len=2000)
            return json.dumps(sanitized, ensure_ascii=False)
        except Exception:
            return str(response)

    # ==================== 连接测试 ====================

    async def test_text_connection(self, model_name: str) -> str:
        """
        发送最小请求测试文本 API 连通性。
        返回: 成功时返回模型回复的前 80 字符；失败时返回错误信息。
        """
        try:
            result = await self.generate_text(
                model_name=model_name,
                contents=[{"type": "text", "text": "Say 'hello' in one word."}],
                system_prompt="",
                temperature=0,
                max_output_tokens=32,
                max_attempts=1,
                retry_delay=0,
            )
            text = result[0] if result else ""
            if text and text != "Error":
                return text[:80]
            return "Error: 未返回有效文本"
        except Exception as e:
            return f"Error: {e}"
        finally:
            # 测试连接通常是一次性调用，主动关闭会话避免 Unclosed client session。
            await self.close()

    async def test_image_connection(self, model_name: str) -> str:
        """
        发送最小请求测试图像 API 连通性。
        返回: 成功时返回 "OK (base64 长度=...)"; 失败时返回错误信息。
        """
        try:
            result = await self.generate_image(
                model_name=model_name,
                prompt="A small red circle on white background",
                aspect_ratio="1:1",
                max_attempts=1,
                retry_delay=0,
            )
            b64 = result[0] if result else ""
            if b64 and b64 != "Error":
                return f"OK (base64 长度={len(b64)})"
            return "Error: 未返回有效图像"
        except Exception as e:
            return f"Error: {e}"
        finally:
            # 测试连接通常是一次性调用，主动关闭会话避免 Unclosed client session。
            await self.close()

    # ==================== 请求头 ====================

    def _openai_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _gemini_headers(self, api_key: str) -> Dict[str, str]:
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

    # ==================== HTTP 请求封装 ====================

    async def _post_json(
        self, url: str, payload: Dict[str, Any], headers: Dict[str, str],
        timeout: int = 120,
    ) -> Dict[str, Any]:
        print(f"[DEBUG] [Multi] POST {url}")
        safe_headers = self._sanitize_headers_for_log(headers)
        safe_payload = self._summarize_for_log(payload)
        print(f"[DEBUG] [Multi]   headers={json.dumps(safe_headers, ensure_ascii=False)}")
        print(f"[DEBUG] [Multi]   payload={json.dumps(safe_payload, ensure_ascii=False)}")
        session = await self._get_session()
        async with session.post(
            url, json=payload, headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            status = resp.status
            body_text = await resp.text()
            try:
                body = json.loads(body_text) if body_text else {}
            except json.JSONDecodeError:
                body = {"raw_response": body_text}
            print(f"[DEBUG] [Multi]   status={status}")
            if status >= 400:
                error_msg = body.get("error", body) if isinstance(body, dict) else body
                print(f"[DEBUG] [Multi]   error: {error_msg}")
                print(f"[DEBUG] [Multi]   error_body={self._dump_response_for_log(body)}")
                if 400 <= status < 500 and status != 429:
                    raise ClientError(f"HTTP {status}: {error_msg}")
            resp.raise_for_status()
            return body

    async def _download_image_as_base64(self, url: str) -> Optional[str]:
        try:
            session = await self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                image_data = await resp.read()
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            print(f"[Multi] 下载图片失败 ({url}): {e}")
            return None

    # ==================== 内容格式转换 ====================

    def _convert_contents_to_openai_messages(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
    ) -> List[Dict[str, Any]]:
        """通用内容列表 -> OpenAI messages 格式（复用 EvolinkProvider 逻辑）"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_parts = []
        has_image = False

        for item in contents:
            item_type = item.get("type", "")
            if item_type == "text":
                user_parts.append({"type": "text", "text": item["text"]})
            elif item_type == "image":
                has_image = True
                source = item.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    user_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    })
                elif "image_base64" in item:
                    user_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{item['image_base64']}"},
                    })

        if not has_image and len(user_parts) == 1:
            messages.append({"role": "user", "content": user_parts[0]["text"]})
        else:
            messages.append({"role": "user", "content": user_parts})

        return messages

    def _convert_contents_to_gemini_parts(
        self,
        contents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """通用内容列表 -> Gemini contents[].parts 格式"""
        parts = []
        for item in contents:
            item_type = item.get("type", "")
            if item_type == "text":
                parts.append({"text": item["text"]})
            elif item_type == "image":
                source = item.get("source", {})
                if source.get("type") == "base64":
                    parts.append({
                        "inline_data": {
                            "mime_type": source.get("media_type", "image/jpeg"),
                            "data": source.get("data", ""),
                        }
                    })
                elif "image_base64" in item:
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": item["image_base64"],
                        }
                    })
        return parts

    # ==================== Gemini 响应提取 ====================

    def _extract_gemini_text(self, json_data: Dict[str, Any]) -> str:
        """从 Gemini 响应中提取文本，过滤 thought: true 部分"""
        try:
            candidates = json_data.get("candidates", [])
            if not candidates:
                return ""
            parts = candidates[0].get("content", {}).get("parts", [])
            texts = []
            for part in parts:
                if part.get("thought", False):
                    continue
                text = part.get("text", "")
                if text:
                    texts.append(text)
            return "".join(texts)
        except Exception as e:
            print(f"[Multi] Gemini 文本提取失败: {e}")
            return ""

    def _extract_gemini_image(self, json_data: Dict[str, Any]) -> Optional[str]:
        """从 Gemini 响应中提取 inlineData 图片，返回 base64"""
        try:
            candidates = json_data.get("candidates", [])
            for candidate in candidates:
                parts = candidate.get("content", {}).get("parts", [])
                for part in parts:
                    inline = part.get("inlineData") or part.get("inline_data")
                    if inline and isinstance(inline.get("data"), str) and inline["data"].strip():
                        return inline["data"].strip()
            return None
        except Exception:
            return None

    # ==================== OpenAI 图像响应提取 ====================

    def _extract_image_from_data_url(self, data_url: str) -> Optional[str]:
        """从 data URL 提取 base64 图片数据。"""
        if not isinstance(data_url, str):
            return None
        match = re.match(
            r"^\s*data:image\/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=]+)\s*$",
            data_url,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return None

    def _extract_http_image_url_from_text(self, text: str) -> Optional[str]:
        """从文本提取 HTTP 图片 URL（markdown/html/plain）。"""
        if not isinstance(text, str):
            return None
        t = text.strip()
        if not t:
            return None

        md_match = re.search(r"!\[[^\]]*]\((https?:\/\/[^)\s]+)\)", t, re.IGNORECASE)
        if md_match:
            return md_match.group(1)

        html_match = re.search(r'<img[^>]+src=["\'](https?:\/\/[^"\']+)["\']', t, re.IGNORECASE)
        if html_match:
            return html_match.group(1)

        plain_match = re.search(r"https?:\/\/[^\s<>()]+", t, re.IGNORECASE)
        if plain_match:
            return plain_match.group(0)

        return None

    def _extract_base64_from_dict(self, item: Dict[str, Any]) -> Optional[str]:
        """从常见图片字段提取 base64。"""
        if not isinstance(item, dict):
            return None
        b64 = (
            item.get("imageBase64")
            or item.get("image_base64")
            or item.get("b64_json")
            or item.get("b64")
            or item.get("result")
        )
        if isinstance(b64, str) and b64.strip():
            return b64.strip()
        return None

    def _extract_image_from_openai_chat_message(self, message: Dict[str, Any]) -> Optional[str]:
        """从 OpenAI chat message/delta 中提取图片 base64。"""
        if not isinstance(message, dict):
            return None

        content = message.get("content")
        images = message.get("images")

        # 1) 字符串 content：json/data-url/纯 base64
        if isinstance(content, str) and content.strip():
            trimmed = content.strip()

            if (trimmed.startswith("{") and trimmed.endswith("}")) or (
                trimmed.startswith("[") and trimmed.endswith("]")
            ):
                try:
                    parsed = json.loads(trimmed)
                    if isinstance(parsed, dict):
                        b64 = self._extract_base64_from_dict(parsed)
                        if b64:
                            return b64
                    elif isinstance(parsed, list):
                        for entry in parsed:
                            if isinstance(entry, dict):
                                b64 = self._extract_base64_from_dict(entry)
                                if b64:
                                    return b64
                except Exception:
                    pass

            data_url_b64 = self._extract_image_from_data_url(trimmed)
            if data_url_b64:
                return data_url_b64

            data_url_match = re.search(
                r"data:image\/[a-zA-Z0-9.+-]+;base64,([A-Za-z0-9+/=]+)",
                content,
                flags=re.IGNORECASE,
            )
            if data_url_match:
                return data_url_match.group(1)

            normalized = re.sub(r"\s+", "", trimmed)
            if (
                len(normalized) > 1024
                and re.match(r"^[A-Za-z0-9+/=]+$", normalized)
                and len(normalized) % 4 == 0
            ):
                return normalized

        # 2) 多模态数组 content
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type", "")).lower()
                if part_type == "image_url":
                    image_url = part.get("image_url") or {}
                    if isinstance(image_url, dict):
                        url = image_url.get("url") or image_url.get("uri")
                    else:
                        url = image_url
                    b64 = self._extract_image_from_data_url(url)
                    if b64:
                        return b64

                if part_type in ("image", "output_image"):
                    b64 = self._extract_base64_from_dict(part)
                    if b64:
                        return b64

        # 3) message.images
        if isinstance(images, list):
            for part in images:
                if not isinstance(part, dict):
                    continue
                image_url = part.get("image_url") or {}
                if isinstance(image_url, dict):
                    url = image_url.get("url") or image_url.get("uri")
                    b64 = self._extract_image_from_data_url(url)
                    if b64:
                        return b64
                b64 = self._extract_base64_from_dict(part)
                if b64:
                    return b64

        return None

    def _extract_image_from_responses_output(self, json_data: Dict[str, Any]) -> Optional[str]:
        """从 OpenAI /v1/responses 响应格式提取图片 base64。"""
        output = json_data.get("output")
        if not isinstance(output, list):
            return None

        for item in output:
            if not isinstance(item, dict):
                continue

            b64 = self._extract_base64_from_dict(item)
            if b64:
                return b64

            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    b64 = self._extract_base64_from_dict(part)
                    if b64:
                        return b64
                    part_type = str(part.get("type", "")).lower()
                    if part_type in ("image_url", "output_image"):
                        image_url = part.get("image_url") or {}
                        if isinstance(image_url, dict):
                            url = image_url.get("url") or image_url.get("uri")
                        else:
                            url = image_url
                        b64 = self._extract_image_from_data_url(url)
                        if b64:
                            return b64

        return None

    def _extract_image_from_openai_response(self, json_data: Dict[str, Any]) -> Optional[str]:
        """
        从 OpenAI 兼容响应中提取图片 base64。
        支持 images/generations、chat/completions、responses 三种格式。
        """
        # 1. images/generations: {data: [{b64_json, url}]}
        if isinstance(json_data.get("data"), list) and json_data["data"]:
            for item in json_data["data"]:
                if not isinstance(item, dict):
                    continue
                b64 = self._extract_base64_from_dict(item)
                if b64:
                    return b64
                url = item.get("url") or item.get("uri") or ""
                b64_from_data_url = self._extract_image_from_data_url(url)
                if b64_from_data_url:
                    return b64_from_data_url

        # 2. chat/completions: {choices: [{message: {content}}]}
        if isinstance(json_data.get("choices"), list):
            for choice in json_data["choices"]:
                msg = choice.get("message") or choice.get("delta") or choice
                b64 = self._extract_image_from_openai_chat_message(msg)
                if b64:
                    return b64

        # 3. responses: {output: [...]}
        b64 = self._extract_image_from_responses_output(json_data)
        if b64:
            return b64

        # 4. Gemini 兼容格式
        if isinstance(json_data.get("candidates"), list) and json_data["candidates"]:
            return self._extract_gemini_image(json_data)

        return None

    def _extract_http_url_from_openai_response(self, json_data: Dict[str, Any]) -> Optional[str]:
        """从 OpenAI 响应中提取 HTTP 图片 URL（用于远程下载）"""
        if isinstance(json_data.get("data"), list) and json_data["data"]:
            for item in json_data["data"]:
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or item.get("uri") or ""
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    return url.strip()

        if isinstance(json_data.get("choices"), list):
            for choice in json_data["choices"]:
                msg = choice.get("message") or choice.get("delta") or choice
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content", "")
                if isinstance(content, str):
                    url = self._extract_http_image_url_from_text(content)
                    if url:
                        return url
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        image_url = part.get("image_url") or {}
                        if isinstance(image_url, dict):
                            url = image_url.get("url") or image_url.get("uri") or ""
                        else:
                            url = image_url
                        if isinstance(url, str) and url.startswith(("http://", "https://")):
                            return url.strip()

                images = msg.get("images")
                if isinstance(images, list):
                    for part in images:
                        if not isinstance(part, dict):
                            continue
                        image_url = part.get("image_url") or {}
                        if isinstance(image_url, dict):
                            url = image_url.get("url") or image_url.get("uri") or ""
                        else:
                            url = image_url
                        if isinstance(url, str) and url.startswith(("http://", "https://")):
                            return url.strip()

        output = json_data.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                url = item.get("url") or item.get("uri") or ""
                if isinstance(url, str) and url.startswith(("http://", "https://")):
                    return url.strip()
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if not isinstance(part, dict):
                            continue
                        image_url = part.get("image_url") or {}
                        if isinstance(image_url, dict):
                            url = image_url.get("url") or image_url.get("uri") or ""
                        else:
                            url = image_url
                        if isinstance(url, str) and url.startswith(("http://", "https://")):
                            return url.strip()
        return None

    # ==================== 文本生成 ====================

    async def generate_text(
        self,
        model_name: str,
        contents: List[Dict[str, Any]],
        system_prompt: str = "",
        temperature: float = 1.0,
        max_output_tokens: int = 50000,
        max_attempts: int = 3,
        retry_delay: float = 5,
        error_context: str = "",
    ) -> List[str]:
        style = self.text_api_style
        api_key = self.text_api_key
        base_url = self.text_base_url

        print(f"[Multi 文本] style={style}, model={model_name}, base_url={base_url}")

        for attempt in range(max_attempts):
            try:
                if style == "openai":
                    text = await self._text_via_openai(
                        model_name, contents, system_prompt,
                        temperature, max_output_tokens, api_key, base_url,
                    )
                else:
                    text = await self._text_via_gemini(
                        model_name, contents, system_prompt,
                        temperature, max_output_tokens, api_key, base_url,
                    )

                if text and text.strip():
                    print(f"[Multi 文本] 成功, 长度={len(text)}")
                    return [text]

                print(f"[Multi 文本] 响应为空，{retry_delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)

            except ClientError as e:
                ctx = f" ({error_context})" if error_context else ""
                print(f"[Multi 文本] 客户端错误{ctx}: {e}。不再重试。")
                return ["Error"]

            except Exception as e:
                ctx = f" ({error_context})" if error_context else ""
                delay = min(retry_delay * (2 ** attempt), 30)
                print(f"[Multi 文本] 第 {attempt + 1} 次失败{ctx}: {e}。{delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                else:
                    print(f"[Multi 文本] 全部 {max_attempts} 次失败{ctx}")

        return ["Error"]

    async def _text_via_openai(
        self, model_name, contents, system_prompt,
        temperature, max_output_tokens, api_key, base_url,
    ) -> str:
        url = f"{base_url}/v1/chat/completions"
        messages = self._convert_contents_to_openai_messages(contents, system_prompt)
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_output_tokens,
        }
        headers = self._openai_headers(api_key)
        response = await self._post_json(url, payload, headers)
        choices = response.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content", "")
        return ""

    async def _text_via_gemini(
        self, model_name, contents, system_prompt,
        temperature, max_output_tokens, api_key, base_url,
    ) -> str:
        url = f"{base_url}/v1beta/models/{model_name}:generateContent"
        parts = self._convert_contents_to_gemini_parts(contents)
        payload: Dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        headers = self._gemini_headers(api_key)
        response = await self._post_json(url, payload, headers, timeout=300)
        return self._extract_gemini_text(response)

    # ==================== 图像生成 ====================

    async def generate_image(
        self,
        model_name: str,
        prompt: str,
        aspect_ratio: str = "16:9",
        quality: str = "2K",
        image_urls: Optional[List[str]] = None,
        max_attempts: int = 3,
        retry_delay: float = 30,
        poll_interval: float = 3,
        error_context: str = "",
    ) -> List[str]:
        style = self.image_api_style
        api_key = self.image_api_key
        base_url = self.image_base_url
        ctx = f" ({error_context})" if error_context else ""

        print(
            f"[Multi 图像] style={style}, model={model_name}, base_url={base_url}, "
            f"openai_endpoint={self.image_openai_endpoint}{ctx}"
        )

        for attempt in range(max_attempts):
            try:
                if style == "openai":
                    b64 = await self._image_via_openai(
                        model_name, prompt, aspect_ratio, quality,
                        api_key, base_url, error_context=error_context,
                    )
                else:
                    b64 = await self._image_via_gemini(
                        model_name, prompt, aspect_ratio, quality,
                        api_key, base_url, error_context=error_context,
                    )

                if b64 and b64 != "Error":
                    return [b64]

                print(f"[Multi 图像] 未获取图像{ctx}，{retry_delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(retry_delay)

            except ClientError as e:
                ctx = f" ({error_context})" if error_context else ""
                print(f"[Multi 图像] 客户端错误{ctx}: {e}。不再重试。")
                return ["Error"]

            except Exception as e:
                ctx = f" ({error_context})" if error_context else ""
                delay = min(retry_delay * (2 ** attempt), 60)
                print(f"[Multi 图像] 第 {attempt + 1} 次失败{ctx}: {e}。{delay}s 后重试...")
                if attempt < max_attempts - 1:
                    await asyncio.sleep(delay)
                else:
                    print(f"[Multi 图像] 全部 {max_attempts} 次失败{ctx}")

        return ["Error"]

    async def _image_via_openai(
        self, model_name, prompt, aspect_ratio, quality,
        api_key, base_url, error_context: str = "",
    ) -> Optional[str]:
        """
        OpenAI 兼容图像生成。
        路由策略由 image_openai_endpoint 控制：
        - chat: 仅 /v1/chat/completions
        - images: 仅 /v1/images/generations
        - auto: 先 chat，失败/无图再回退 images
        """
        headers = self._openai_headers(api_key)
        ctx = f" ({error_context})" if error_context else ""

        route_mode = self.image_openai_endpoint
        if route_mode in ("chat", "auto"):
            url = f"{base_url}/v1/chat/completions"
            system_parts = ["You are an image generation model. Return a single image (no text)."]
            if aspect_ratio:
                system_parts.append(f"Aspect ratio: {aspect_ratio}.")
            if quality:
                system_parts.append(f"Resolution: {quality}.")
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": " ".join(system_parts)},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.8,
                "stream": False,
            }
            try:
                response = await self._post_json(url, payload, headers, timeout=300)
                b64 = self._extract_image_from_openai_response(response)
                if b64:
                    return b64
                remote_url = self._extract_http_url_from_openai_response(response)
                if remote_url:
                    downloaded = await self._download_image_as_base64(remote_url)
                    if downloaded:
                        return downloaded
                    print(
                        f"[Multi 图像] chat/completions 返回了 URL 但下载失败{ctx}，响应正文: "
                        f"{self._dump_response_for_log(response)}"
                    )
                    return None
                print(
                    f"[Multi 图像] chat/completions 未提取到图片{ctx}，响应正文: "
                    f"{self._dump_response_for_log(response)}"
                )
            except ClientError:
                raise
            except Exception as e:
                if route_mode == "auto":
                    print(f"[Multi 图像] chat/completions 失败{ctx}: {e}，尝试 images/generations...")
                else:
                    print(f"[Multi 图像] chat/completions 失败{ctx}: {e}")
            if route_mode == "chat":
                return None

        if route_mode in ("images", "auto"):
            url = f"{base_url}/v1/images/generations"
            payload = {
                "model": model_name,
                "prompt": prompt,
                "response_format": "b64_json",
            }
            response = await self._post_json(url, payload, headers, timeout=300)
            b64 = self._extract_image_from_openai_response(response)
            if b64:
                return b64
            remote_url = self._extract_http_url_from_openai_response(response)
            if remote_url:
                downloaded = await self._download_image_as_base64(remote_url)
                if downloaded:
                    return downloaded
                print(
                    f"[Multi 图像] images/generations 返回了 URL 但下载失败{ctx}，响应正文: "
                    f"{self._dump_response_for_log(response)}"
                )
                return None
            print(
                f"[Multi 图像] images/generations 未提取到图片{ctx}，响应正文: "
                f"{self._dump_response_for_log(response)}"
            )
        return None

    async def _image_via_gemini(
        self, model_name, prompt, aspect_ratio, quality,
        api_key, base_url, error_context: str = "",
    ) -> Optional[str]:
        """Gemini 原生图像生成（responseModalities: IMAGE）"""
        ctx = f" ({error_context})" if error_context else ""
        url = f"{base_url}/v1beta/models/{model_name}:generateContent"
        generation_config: Dict[str, Any] = {
            "temperature": 0.8,
            "responseModalities": ["TEXT", "IMAGE"],
        }
        image_config: Dict[str, Any] = {}
        if aspect_ratio:
            image_config["aspectRatio"] = aspect_ratio
        if quality:
            image_config["imageSize"] = quality
        if image_config:
            generation_config["imageConfig"] = image_config

        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        headers = self._gemini_headers(api_key)
        response = await self._post_json(url, payload, headers, timeout=300)
        b64 = self._extract_gemini_image(response)
        if not b64:
            print(
                f"[Multi 图像] Gemini 响应未包含可用图片{ctx}，响应正文: "
                f"{self._dump_response_for_log(response)}"
            )
        return b64
