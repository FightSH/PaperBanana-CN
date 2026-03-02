import asyncio
import re
import base64
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import aiohttp  # 异步 HTTP 请求库（需安装：pip install aiohttp）


# ======================== 数据结构定义 ========================
@dataclass
class ImageGenerationResult:
    """图片生成结果"""
    image_base64: str  # Base64 编码的图片数据
    mime_type: str  # 图片 MIME 类型（如 image/png）


class ImageGenerationError(Exception):
    """图片生成异常类"""

    def __init__(self, message: str, details: Dict[str, Any]):
        super().__init__(message)
        self.details = details  # 详细错误信息
        self.message = message


# ======================== 核心工具函数 ========================
def is_sensitive_key(key: Any) -> bool:
    """判断字段名是否包含敏感信息。"""
    if not isinstance(key, str):
        return False
    lower = key.lower()
    return any(token in lower for token in ("key", "token", "secret", "authorization", "password"))


def looks_like_base64(value: Any) -> bool:
    """粗略判断字符串是否像 base64（用于日志脱敏）。"""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    if len(stripped) < 512:
        return False
    if stripped.startswith("data:") and ";base64," in stripped:
        return True
    return bool(re.match(r"^[A-Za-z0-9+/=\s]+$", stripped))


def sanitize_for_log(value: Any, parent_key: str = "", max_str_len: int = 1200) -> Any:
    """递归脱敏并裁剪日志内容，避免泄露密钥和刷屏。"""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if is_sensitive_key(k):
                out[k] = "***"
            else:
                out[k] = sanitize_for_log(v, parent_key=str(k), max_str_len=max_str_len)
        return out
    if isinstance(value, list):
        return [sanitize_for_log(v, parent_key=parent_key, max_str_len=max_str_len) for v in value]
    if isinstance(value, str):
        if value.startswith("data:") and ";base64," in value:
            prefix, b64 = value.split(";base64,", 1)
            return f"{prefix};base64,<omitted len={len(b64)}>"
        if is_sensitive_key(parent_key):
            return "***"
        if looks_like_base64(value):
            compact = re.sub(r"\s+", "", value)
            return f"<base64 omitted len={len(compact)}>"
        if len(value) > max_str_len:
            return f"{value[:max_str_len]}...(len={len(value)})"
    return value


def dump_for_log(value: Any) -> str:
    """将任意响应对象转换为可打印的脱敏文本。"""
    try:
        return json.dumps(sanitize_for_log(value), ensure_ascii=False)
    except Exception:
        return str(value)


def resolve_request_mode(value: Any) -> str:
    """解析请求模式（gemini/openai）"""
    raw = str(value or "").strip().lower()
    if raw in ["openai", "openai-compat"]:
        return "openai"
    return "gemini"


def normalize_api_url(url: str) -> str:
    """标准化 API 地址（去除末尾斜杠）"""
    return url.strip().rstrip("/") if url else ""


def extract_image_from_data_url(data_url: str) -> Optional[Dict[str, str]]:
    """从 Data URL 中提取 Base64 图片和 MIME 类型"""
    pattern = r"^data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)\s*$"
    match = re.match(pattern, data_url, re.IGNORECASE)
    if match:
        return {
            "mime_type": match.group(1),
            "image_base64": match.group(2)
        }
    return None


def guess_mime_type_from_url(url: str) -> Optional[str]:
    """从 URL 后缀猜测 MIME 类型"""
    clean = re.split(r"[?#]", url.lower(), maxsplit=1)[0] if url else ""
    if clean.endswith(".png"):
        return "image/png"
    elif clean.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    elif clean.endswith(".webp"):
        return "image/webp"
    elif clean.endswith(".gif"):
        return "image/gif"
    elif clean.endswith(".svg"):
        return "image/svg+xml"
    return None


def extract_http_image_url_from_text(text: str) -> Optional[str]:
    """从文本中提取 HTTP 图片 URL（支持 Markdown/HTML/纯文本）"""
    t = text.strip() if text else ""
    if not t:
        return None

    # 匹配 Markdown 图片: ![alt](url)
    md_match = re.search(r"!\[[^\]]*]\((https?:\/\/[^)\s]+)\)", t, re.IGNORECASE)
    if md_match:
        return md_match.group(1)

    # 匹配 HTML 图片: <img src="url">
    html_match = re.search(r'<img[^>]+src=["\'](https?:\/\/[^"\']+)["\']', t, re.IGNORECASE)
    if html_match:
        return html_match.group(1)

    # 匹配纯 URL
    plain_match = re.search(r"https?:\/\/[^\s<>()]+", t, re.IGNORECASE)
    if plain_match:
        return plain_match.group(0)

    return None


async def download_image_url_as_base64(session: aiohttp.ClientSession, url: str) -> ImageGenerationResult:
    """下载远程图片并转为 Base64"""
    endpoint = url.strip()
    if not endpoint:
        raise ImageGenerationError("空图片URL", {
            "errorName": "ImageDownloadError",
            "errorMessage": "Empty image URL",
            "requestUrl": endpoint
        })

    try:
        async with session.get(
                endpoint,
                headers={"Accept": "image/*,*/*;q=0.8"},
                timeout=aiohttp.ClientTimeout(total=180)  # 3分钟超时
        ) as response:
            if response.status != 200:
                raise ImageGenerationError(f"HTTP {response.status}", {
                    "errorName": f"HTTP_{response.status}",
                    "errorMessage": f"HTTP {response.status}: {response.reason}",
                    "statusCode": response.status,
                    "requestUrl": endpoint
                })

            # 获取 MIME 类型
            content_type = response.headers.get("Content-Type", "")
            header_mime = content_type.split(";")[0].strip() if content_type else ""
            mime_type = header_mime or guess_mime_type_from_url(endpoint) or "image/jpeg"

            # 读取二进制数据并转为 Base64
            image_data = await response.read()
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            return ImageGenerationResult(
                image_base64=image_base64,
                mime_type=mime_type
            )
    except ImageGenerationError:
        raise
    except Exception as e:
        raise ImageGenerationError("下载图片失败", {
            "errorName": "ImageDownloadError",
            "errorMessage": str(e),
            "requestUrl": endpoint
        })


# ======================== OpenAI 兼容 API 处理 ========================
def extract_image_from_openai_chat_message(message: Any) -> Optional[Dict[str, str]]:
    """从 OpenAI 聊天消息中提取图片数据"""
    content = message.get("content")
    images = message.get("images")

    # 处理字符串类型的 content
    if isinstance(content, str) and content.strip():
        trimmed = content.strip()

        # 尝试解析 JSON 格式
        if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("[") and trimmed.endswith("]")):
            try:
                parsed = json.loads(trimmed)
                parsed_items = parsed if isinstance(parsed, list) else [parsed]
                for item in parsed_items:
                    if not isinstance(item, dict):
                        continue
                    b64 = (
                        item.get("imageBase64")
                        or item.get("image_base64")
                        or item.get("b64_json")
                        or item.get("b64")
                    )
                    if isinstance(b64, str) and b64.strip():
                        mime = item.get("mimeType") or item.get("mime_type") or "image/png"
                        return {
                            "mime_type": mime,
                            "image_base64": b64.strip()
                        }
            except:
                pass

        # 尝试提取 Data URL
        data_url_match = extract_image_from_data_url(trimmed)
        if data_url_match:
            return data_url_match

        # 匹配文本中的 Data URL
        data_url_pattern = r"data:(image\/[a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=]+)"
        data_match = re.search(data_url_pattern, content, re.IGNORECASE)
        if data_match:
            return {
                "mime_type": data_match.group(1),
                "image_base64": data_match.group(2)
            }

        # 兜底：纯 Base64 字符串
        normalized = re.sub(r"\s+", "", trimmed)
        if (len(normalized) > 1024 and
                re.match(r"^[A-Za-z0-9+/=]+$", normalized) and
                len(normalized) % 4 == 0):
            return {
                "mime_type": "image/png",
                "image_base64": normalized
            }

    # 处理数组类型的 content（多模态）
    if isinstance(content, list):
        for part in content:
            part_type = str(part.get("type", "")).lower()

            # 处理 image_url 类型
            if part_type == "image_url":
                url = part.get("image_url", {}).get("url") or part.get("image_url", {}).get("uri")
                if isinstance(url, str):
                    data_url_match = extract_image_from_data_url(url.strip())
                    if data_url_match:
                        return data_url_match

            # 处理 image 类型
            if part_type in ["image", "output_image"]:
                b64 = part.get("image_base64") or part.get("b64_json") or part.get("b64")
                if isinstance(b64, str) and b64.strip():
                    mime = part.get("mime_type") or part.get("mimeType") or "image/png"
                    return {
                        "mime_type": mime,
                        "image_base64": b64.strip()
                    }

    # 处理 message.images 字段
    if isinstance(images, list):
        for part in images:
            url = part.get("image_url", {}).get("url") or part.get("image_url", {}).get("uri")
            if isinstance(url, str):
                data_url_match = extract_image_from_data_url(url.strip())
                if data_url_match:
                    return data_url_match

    return None


def extract_http_image_url_from_openai_response(json_data: Any) -> Optional[str]:
    """从 OpenAI 响应中提取 HTTP 图片 URL"""
    # 1. 处理 images/generations 格式: {data: [{url}]}
    if isinstance(json_data.get("data"), list) and len(json_data["data"]) > 0:
        item = next((d for d in json_data["data"] if d.get("url")), json_data["data"][0])
        url = item.get("url")
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            return url.strip()

    # 2. 处理 chat/completions 格式: {choices: [{message}]}
    if isinstance(json_data.get("choices"), list):
        for choice in json_data["choices"]:
            msg = choice.get("message") or choice.get("delta") or choice
            # 提取 message 中的 URL
            content = msg.get("content")
            if isinstance(content, str):
                url = extract_http_image_url_from_text(content)
                if url and url.startswith(("http://", "https://")):
                    return url
            if isinstance(content, list):
                for part in content:
                    url = part.get("image_url", {}).get("url") or part.get("url")
                    if isinstance(url, str) and url.startswith(("http://", "https://")):
                        return url.strip()

    return None


def extract_image_from_openai_response(json_data: Any) -> Optional[Dict[str, str]]:
    """从 OpenAI 响应中提取图片数据"""
    # 1. 处理 images/generations 格式
    if isinstance(json_data.get("data"), list) and len(json_data["data"]) > 0:
        item = next((d for d in json_data["data"] if d.get("b64_json") or d.get("b64")), json_data["data"][0])
        b64 = item.get("b64_json") or item.get("b64")
        if isinstance(b64, str) and b64.strip():
            return {
                "mime_type": "image/png",
                "image_base64": b64.strip()
            }
        # 尝试 Data URL
        url = item.get("url")
        if isinstance(url, str):
            data_url_match = extract_image_from_data_url(url.strip())
            if data_url_match:
                return data_url_match

    # 2. 处理 chat/completions 格式
    if isinstance(json_data.get("choices"), list):
        for choice in json_data["choices"]:
            msg = choice.get("message") or choice.get("delta") or choice
            image_data = extract_image_from_openai_chat_message(msg)
            if image_data:
                return image_data

    # 3. 兼容 Gemini 格式
    if isinstance(json_data.get("candidates"), list) and len(json_data["candidates"]) > 0:
        parts = json_data["candidates"][0].get("content", {}).get("parts", [])
        image_part = next((p for p in parts if p.get("inlineData")), None)
        if image_part and image_part["inlineData"].get("data"):
            return {
                "mime_type": image_part["inlineData"].get("mimeType") or "image/png",
                "image_base64": image_part["inlineData"]["data"]
            }

    return None


async def generate_image_via_openai(
        session: aiohttp.ClientSession,
        prompt: str,
        api_key: str,
        api_url: str,
        model: str,
        aspect_ratio: str = "",
        resolution: str = ""
) -> ImageGenerationResult:
    """调用 OpenAI 兼容 API 生成图片"""
    api_url = normalize_api_url(api_url)
    model = model.strip()

    # 自动补全端点路径
    if not re.search(
            r"(\/v1\/(chat\/completions|responses|images\/generations)\b|\/(chat\/completions|responses|images\/generations)\b)",
            api_url, re.IGNORECASE):
        api_url = f"{api_url}/v1/chat/completions" if not api_url.endswith("/v1") else f"{api_url}/chat/completions"

    # 判断端点类型
    is_images_endpoint = re.search(r"\/(v1\/)?images\/generations\b", api_url, re.IGNORECASE) and not re.search(
        r"\/chat\/completions\b", api_url, re.IGNORECASE)
    is_responses_endpoint = re.search(r"\/(v1\/)?responses\b", api_url, re.IGNORECASE) and not re.search(
        r"\/chat\/completions\b", api_url, re.IGNORECASE)

    # 构造请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # 构造请求体
    if is_images_endpoint:
        payload = {
            "model": model,
            "prompt": prompt,
            "response_format": "b64_json"
        }
    elif is_responses_endpoint:
        # 构建 system prompt
        system_parts = ["You are an image generation model. Return a single image (no text)."]
        if aspect_ratio:
            system_parts.append(f"Aspect ratio: {aspect_ratio}.")
        if resolution:
            system_parts.append(f"Resolution: {resolution}.")

        payload = {
            "model": model,
            "input": [
                {"role": "system", "content": " ".join(system_parts)},
                {"role": "user", "content": prompt}
            ]
        }
    else:
        # chat/completions 格式
        system_parts = ["You are an image generation model. Return a single image (no text)."]
        if aspect_ratio:
            system_parts.append(f"Aspect ratio: {aspect_ratio}.")
        if resolution:
            system_parts.append(f"Resolution: {resolution}.")

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": " ".join(system_parts)},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.8,
            "stream": False
        }

    # 发送请求
    try:
        async with session.post(
                api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
        ) as response:
            response_text = await response.text()
            try:
                json_data = json.loads(response_text)
            except json.JSONDecodeError:
                json_data = {"raw_response": response_text}

            if response.status != 200:
                # 解析错误信息
                error_details = {}
                try:
                    error_json = json_data
                    err = error_json.get("error") or error_json
                    error_details = {
                        "errorName": err.get("code") or err.get("type") or f"HTTP_{response.status}",
                        "errorMessage": err.get("message") or f"HTTP {response.status}: {response.reason}",
                        "statusCode": response.status,
                        "requestUrl": api_url,
                        "responseBody": response_text[:1000]  # 截断长响应
                    }
                except:
                    error_details = {
                        "errorName": f"HTTP_{response.status}",
                        "errorMessage": f"HTTP {response.status}: {response.reason}",
                        "statusCode": response.status,
                        "requestUrl": api_url,
                        "responseBody": response_text[:1000]
                    }
                print(f"[ImageClient][OpenAI] ❌ status={response.status}, response={dump_for_log(json_data)}")
                raise ImageGenerationError(error_details["errorMessage"], error_details)

            # 提取图片数据
            image_data = extract_image_from_openai_response(json_data)
            if image_data:
                return ImageGenerationResult(
                    image_base64=image_data["image_base64"],
                    mime_type=image_data["mime_type"]
                )

            # 尝试提取远程图片 URL 并下载
            remote_url = extract_http_image_url_from_openai_response(json_data)
            if remote_url:
                return await download_image_url_as_base64(session, remote_url)

            # 无图片数据
            print(f"[ImageClient][OpenAI] ⚠️ 未提取到图片，response={dump_for_log(json_data)}")
            raise ImageGenerationError("API 未返回图片数据", {
                "errorName": "NoImageData",
                "errorMessage": "OpenAI 兼容接口响应中未识别到图片数据",
                "requestUrl": api_url,
                "responseBody": response_text[:800]
            })

    except ImageGenerationError:
        raise
    except Exception as e:
        # 网络错误处理
        raise ImageGenerationError("OpenAI 兼容生图请求失败", {
            "errorName": "NetworkError",
            "errorMessage": str(e),
            "requestUrl": api_url
        })


# ======================== Gemini API 处理 ========================
async def generate_image_via_gemini(
        session: aiohttp.ClientSession,
        prompt: str,
        api_key: str,
        api_url: str,
        model: str,
        aspect_ratio: str = "",
        resolution: str = ""
) -> ImageGenerationResult:
    """调用 Gemini API 生成图片"""
    # 构造请求端点
    api_url = normalize_api_url(api_url)
    endpoint = f"{api_url}/v1beta/models/{model}:generateContent"

    # 构造 generationConfig
    generation_config = {
        "temperature": 0.8,
        "responseModalities": ["TEXT", "IMAGE"]
    }

    # 添加图片配置（仅当参数启用时）
    image_config = {}
    if aspect_ratio:
        image_config["aspectRatio"] = aspect_ratio
    if resolution:
        image_config["imageSize"] = resolution
    if image_config:
        generation_config["imageConfig"] = image_config

    # 构造请求体
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": generation_config
    }

    # 构造请求头
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    # 发送请求
    try:
        async with session.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload),
                timeout=aiohttp.ClientTimeout(total=300)  # 5分钟超时
        ) as response:
            response_text = await response.text()
            try:
                json_data = json.loads(response_text)
            except json.JSONDecodeError:
                json_data = {"raw_response": response_text}

            if response.status != 200:
                # 解析错误信息
                error_details = {}
                try:
                    error_json = json_data
                    err = error_json.get("error") or error_json
                    error_details = {
                        "errorName": err.get("code") or err.get("status") or f"HTTP_{response.status}",
                        "errorMessage": err.get("message") or f"HTTP {response.status}: {response.reason}",
                        "statusCode": response.status,
                        "requestUrl": endpoint,
                        "responseBody": response_text[:1000]
                    }
                except:
                    error_details = {
                        "errorName": f"HTTP_{response.status}",
                        "errorMessage": f"HTTP {response.status}: {response.reason}",
                        "statusCode": response.status,
                        "requestUrl": endpoint,
                        "responseBody": response_text[:1000]
                    }
                print(f"[ImageClient][Gemini] ❌ status={response.status}, response={dump_for_log(json_data)}")
                raise ImageGenerationError(error_details["errorMessage"], error_details)

            # 提取图片数据
            candidates = json_data.get("candidates", [])
            if not candidates:
                print(f"[ImageClient][Gemini] ⚠️ candidates 为空，response={dump_for_log(json_data)}")
                raise ImageGenerationError("API 未返回任何结果", {
                    "errorName": "EmptyResponse",
                    "errorMessage": "Gemini API 返回了空的 candidates 数组",
                    "requestUrl": endpoint,
                    "responseBody": response_text[:800]
                })

            parts = candidates[0].get("content", {}).get("parts", [])
            image_part = next((p for p in parts if p.get("inlineData")), None)

            if not image_part:
                # 检查是否返回文本
                text_part = next((p for p in parts if p.get("text")), None)
                if text_part:
                    print(f"[ImageClient][Gemini] ⚠️ 返回文本无图片，response={dump_for_log(json_data)}")
                    raise ImageGenerationError("API 返回了文本而非图片", {
                        "errorName": "NoImageGenerated",
                        "errorMessage": f"模型返回了文本内容而非图片：{text_part['text'][:200]}...",
                        "requestUrl": endpoint,
                        "responseBody": response_text[:800]
                    })
                print(f"[ImageClient][Gemini] ⚠️ 无 inlineData 图片，response={dump_for_log(json_data)}")
                raise ImageGenerationError("API 未返回图片数据", {
                    "errorName": "NoImageData",
                    "errorMessage": "Gemini API 响应中未包含 inlineData 图片数据",
                    "requestUrl": endpoint,
                    "responseBody": response_text[:800]
                })

            # 提取 Base64 和 MIME 类型
            image_base64 = image_part["inlineData"]["data"]
            mime_type = image_part["inlineData"].get("mimeType") or "image/png"

            return ImageGenerationResult(
                image_base64=image_base64,
                mime_type=mime_type
            )

    except ImageGenerationError:
        raise
    except Exception as e:
        # 网络错误处理
        raise ImageGenerationError("Gemini 生图请求失败", {
            "errorName": "NetworkError",
            "errorMessage": str(e),
            "requestUrl": endpoint
        })


# ======================== 主调用类 ========================
class ImageClient:
    """图片生成客户端（兼容 Gemini/OpenAI 接口）"""

    @staticmethod
    async def generate_image(
            prompt: str,
            api_key: str,
            request_mode: str = "gemini",
            api_url: Optional[str] = None,
            model: Optional[str] = None,
            aspect_ratio: str = "",
            resolution: str = "",
            aspect_ratio_enabled: bool = False,
            resolution_enabled: bool = False
    ) -> ImageGenerationResult:
        """
        生成学术概念海报图片

        参数:
            prompt: 生图提示词
            api_key: API 密钥
            request_mode: 请求模式（gemini/openai）
            api_url: API 地址（默认：Gemini -> https://generativelanguage.googleapis.com；OpenAI -> https://api.openai.com/v1/chat/completions）
            model: 模型名称（默认：gemini-3-pro-image-preview）
            aspect_ratio: 图片宽高比（如 16:9）
            resolution: 图片分辨率（如 1K）
            aspect_ratio_enabled: 是否启用宽高比参数
            resolution_enabled: 是否启用分辨率参数
        """
        # 解析请求模式
        request_mode = resolve_request_mode(request_mode)

        # 默认配置
        default_api_url = {
            "gemini": "https://generativelanguage.googleapis.com",
            "openai": "https://api.openai.com/v1/chat/completions"
        }
        default_model = "gemini-3-pro-image-preview"

        # 补全参数
        api_url = api_url or default_api_url[request_mode]
        model = model or default_model

        # 过滤未启用的参数
        aspect_ratio = aspect_ratio if aspect_ratio_enabled else ""
        resolution = resolution if resolution_enabled else ""

        # 校验 API Key
        if not api_key:
            raise ImageGenerationError("API Key 未配置", {
                "errorName": "ConfigurationError",
                "errorMessage": f"请配置 {request_mode.upper()} API Key"
            })

        # 创建异步会话
        async with aiohttp.ClientSession() as session:
            if request_mode == "openai":
                return await generate_image_via_openai(
                    session=session,
                    prompt=prompt,
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution
                )
            else:
                return await generate_image_via_gemini(
                    session=session,
                    prompt=prompt,
                    api_key=api_key,
                    api_url=api_url,
                    model=model,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution
                )

    @staticmethod
    async def test_connection(
            api_key: str,
            request_mode: str = "gemini",
            api_url: Optional[str] = None,
            model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        测试 API 连接

        返回:
            {
                "success": bool,
                "message": str
            }
        """
        test_prompt = "Generate a simple test image: a blue circle on white background."
        try:
            result = await ImageClient.generate_image(
                prompt=test_prompt,
                api_key=api_key,
                request_mode=request_mode,
                api_url=api_url,
                model=model
            )
            if result.image_base64:
                size_kb = len(result.image_base64) // 1024
                return {
                    "success": True,
                    "message": f"✅ 连接成功！生成了 {result.mime_type} 格式的图片 ({size_kb} KB)"
                }
            else:
                return {
                    "success": False,
                    "message": "⚠️ 连接成功但未返回图片数据"
                }
        except ImageGenerationError as e:
            return {
                "success": False,
                "message": f"❌ 连接失败: {e.message}"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"❌ 连接失败: {str(e)}"
            }

    @staticmethod
    def format_error(error: ImageGenerationError) -> str:
        """格式化错误信息（用于 UI 显示）"""
        details = error.details
        report = f"错误名称: {details.get('errorName', 'Unknown')}\n"
        report += f"错误信息: {details.get('errorMessage', error.message)}\n"
        if details.get("statusCode"):
            report += f"HTTP 状态码: {details['statusCode']}\n"
        if details.get("requestUrl"):
            report += f"请求地址: {details['requestUrl']}\n"
        if details.get("responseBody"):
            report += f"\n响应内容:\n{details['responseBody']}"
            if len(details['responseBody']) > 1000:
                report += "\n... (已截断)"
        return report


# ======================== 使用示例 ========================
async def main():
    """使用示例"""
    # -------------------- 1. Gemini API 调用示例 --------------------
    try:
        # 配置
        gemini_api_key = "你的 Gemini API Key"  # 替换为实际 Key
        gemini_prompt = "生成一张关于 Python 异步编程的学术概念海报，风格简洁，配色清新，16:9 比例"

        # 生成图片
        gemini_result = await ImageClient.generate_image(
            prompt=gemini_prompt,
            api_key=gemini_api_key,
            request_mode="gemini",
            aspect_ratio="16:9",
            aspect_ratio_enabled=True,
            resolution="1K",
            resolution_enabled=True
        )

        # 保存图片
        with open("gemini_image.png", "wb") as f:
            f.write(base64.b64decode(gemini_result.image_base64))
        print("Gemini 图片已保存为 gemini_image.png")

    except ImageGenerationError as e:
        print("Gemini 调用失败:")
        print(ImageClient.format_error(e))

    # -------------------- 2. OpenAI 兼容 API 调用示例 --------------------
    try:
        # 配置
        openai_api_key = "你的 OpenAI API Key"  # 替换为实际 Key
        openai_api_url = "https://api.openai.com/v1/images/generations"  # 或自定义代理地址
        openai_prompt = "生成一张关于机器学习分类算法的学术海报，专业风格，高对比度"

        # 生成图片
        openai_result = await ImageClient.generate_image(
            prompt=openai_prompt,
            api_key=openai_api_key,
            request_mode="openai",
            api_url=openai_api_url,
            model="dall-e-3"
        )

        # 保存图片
        with open("openai_image.png", "wb") as f:
            f.write(base64.b64decode(openai_result.image_base64))
        print("OpenAI 图片已保存为 openai_image.png")

    except ImageGenerationError as e:
        print("OpenAI 调用失败:")
        print(ImageClient.format_error(e))

    # -------------------- 3. 测试 API 连接 --------------------
    test_result = await ImageClient.test_connection(
        api_key=gemini_api_key,
        request_mode="gemini"
    )
    print(f"\n连接测试结果: {test_result['message']}")


# 运行示例
if __name__ == "__main__":
    asyncio.run(main())
