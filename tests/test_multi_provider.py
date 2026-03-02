"""
MultiProvider 单元测试
测试文本和图像分别路由到不同 API 后端的核心逻辑
"""

import asyncio
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from io import BytesIO
from PIL import Image

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers.multi import MultiProvider


# ==================== 辅助函数 ====================

def make_png_base64():
    """创建一个最小的 PNG 图片并返回 base64 字符串"""
    img = Image.new("RGB", (10, 10), color="blue")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def make_provider(
    text_api_style="openai",
    text_api_key="text-key",
    text_base_url="https://text-api.example.com",
    image_api_style="gemini",
    image_api_key="image-key",
    image_base_url="https://image-api.example.com",
):
    return MultiProvider(
        text_api_style=text_api_style,
        text_api_key=text_api_key,
        text_base_url=text_base_url,
        image_api_style=image_api_style,
        image_api_key=image_api_key,
        image_base_url=image_base_url,
    )


def run(coro):
    """在同步测试中运行 async 协程"""
    return asyncio.get_event_loop().run_until_complete(coro)


# ==================== 初始化测试 ====================

class TestMultiProviderInit:
    def test_init_stores_params(self):
        p = make_provider()
        assert p.text_api_style == "openai"
        assert p.text_api_key == "text-key"
        assert p.text_base_url == "https://text-api.example.com"
        assert p.image_api_style == "gemini"
        assert p.image_api_key == "image-key"
        assert p.image_base_url == "https://image-api.example.com"

    def test_init_strips_trailing_slash(self):
        p = MultiProvider(
            text_api_style="openai",
            text_api_key="k1",
            text_base_url="https://api.example.com/",
            image_api_style="gemini",
            image_api_key="k2",
            image_base_url="https://gemini.example.com/",
        )
        assert p.text_base_url == "https://api.example.com"
        assert p.image_base_url == "https://gemini.example.com"

    def test_headers_openai(self):
        p = make_provider()
        headers = p._openai_headers("test-key-123")
        assert headers["Authorization"] == "Bearer test-key-123"
        assert headers["Content-Type"] == "application/json"

    def test_headers_gemini(self):
        p = make_provider()
        headers = p._gemini_headers("gemini-key-456")
        assert headers["x-goog-api-key"] == "gemini-key-456"
        assert headers["Content-Type"] == "application/json"


class TestConnectionHelpers:
    def test_test_text_connection_always_closes_session_on_success(self):
        p = make_provider()
        with patch.object(p, "generate_text", new_callable=AsyncMock, return_value=["hello"]), \
             patch.object(p, "close", new_callable=AsyncMock) as mock_close:
            result = run(p.test_text_connection("test-model"))
        assert result == "hello"
        mock_close.assert_awaited_once()

    def test_test_text_connection_always_closes_session_on_error(self):
        p = make_provider()
        with patch.object(p, "generate_text", new_callable=AsyncMock, side_effect=Exception("network down")), \
             patch.object(p, "close", new_callable=AsyncMock) as mock_close:
            result = run(p.test_text_connection("test-model"))
        assert result.startswith("Error:")
        mock_close.assert_awaited_once()

    def test_test_image_connection_always_closes_session_on_success(self):
        p = make_provider()
        with patch.object(p, "generate_image", new_callable=AsyncMock, return_value=["abc123"]), \
             patch.object(p, "close", new_callable=AsyncMock) as mock_close:
            result = run(p.test_image_connection("test-model"))
        assert result == "OK (base64 长度=6)"
        mock_close.assert_awaited_once()

    def test_test_image_connection_always_closes_session_on_error(self):
        p = make_provider()
        with patch.object(p, "generate_image", new_callable=AsyncMock, side_effect=Exception("network down")), \
             patch.object(p, "close", new_callable=AsyncMock) as mock_close:
            result = run(p.test_image_connection("test-model"))
        assert result.startswith("Error:")
        mock_close.assert_awaited_once()


# ==================== 内容格式转换测试 ====================

class TestContentConversion:
    def test_openai_text_only(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hello"}]
        messages = p._convert_contents_to_openai_messages(contents, system_prompt="Be helpful")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "Be helpful"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello"

    def test_openai_text_and_image(self):
        p = make_provider()
        img_b64 = make_png_base64()
        contents = [
            {"type": "text", "text": "Describe"},
            {"type": "image", "source": {"type": "base64", "data": img_b64, "media_type": "image/png"}},
        ]
        messages = p._convert_contents_to_openai_messages(contents, system_prompt="")
        assert messages[0]["role"] == "user"
        user_content = messages[0]["content"]
        assert isinstance(user_content, list)
        types = {part["type"] for part in user_content}
        assert "text" in types
        assert "image_url" in types

    def test_openai_image_base64_shorthand(self):
        p = make_provider()
        img_b64 = make_png_base64()
        contents = [
            {"type": "text", "text": "Look"},
            {"type": "image", "image_base64": img_b64},
        ]
        messages = p._convert_contents_to_openai_messages(contents, system_prompt="")
        user_content = messages[0]["content"]
        image_parts = [x for x in user_content if x["type"] == "image_url"]
        assert len(image_parts) == 1
        assert image_parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_gemini_text_only(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hello Gemini"}]
        parts = p._convert_contents_to_gemini_parts(contents)
        assert len(parts) == 1
        assert parts[0] == {"text": "Hello Gemini"}

    def test_gemini_text_and_image(self):
        p = make_provider()
        img_b64 = make_png_base64()
        contents = [
            {"type": "text", "text": "Analyze"},
            {"type": "image", "source": {"type": "base64", "data": img_b64, "media_type": "image/png"}},
        ]
        parts = p._convert_contents_to_gemini_parts(contents)
        assert len(parts) == 2
        assert "text" in parts[0]
        assert "inline_data" in parts[1]
        assert parts[1]["inline_data"]["mime_type"] == "image/png"

    def test_empty_system_prompt_skipped(self):
        p = make_provider()
        contents = [{"type": "text", "text": "Hi"}]
        messages = p._convert_contents_to_openai_messages(contents, system_prompt="")
        assert messages[0]["role"] == "user"


# ==================== Gemini 响应提取测试 ====================

class TestGeminiExtraction:
    def test_extract_text(self):
        p = make_provider()
        data = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello from Gemini"}]
                }
            }]
        }
        assert p._extract_gemini_text(data) == "Hello from Gemini"

    def test_extract_text_filters_thought(self):
        p = make_provider()
        data = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "thinking...", "thought": True},
                        {"text": "Final answer"},
                    ]
                }
            }]
        }
        assert p._extract_gemini_text(data) == "Final answer"

    def test_extract_text_empty_candidates(self):
        p = make_provider()
        assert p._extract_gemini_text({"candidates": []}) == ""
        assert p._extract_gemini_text({}) == ""

    def test_extract_image(self):
        p = make_provider()
        img_b64 = make_png_base64()
        data = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Here is the image"},
                        {"inlineData": {"data": img_b64, "mimeType": "image/png"}},
                    ]
                }
            }]
        }
        result = p._extract_gemini_image(data)
        assert result == img_b64

    def test_extract_image_inline_data_snake_case(self):
        p = make_provider()
        img_b64 = make_png_base64()
        data = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"inline_data": {"data": img_b64, "mimeType": "image/png"}},
                    ]
                }
            }]
        }
        result = p._extract_gemini_image(data)
        assert result == img_b64

    def test_extract_image_from_second_candidate(self):
        p = make_provider()
        img_b64 = make_png_base64()
        data = {
            "candidates": [
                {"content": {"parts": [{"text": "only text"}]}},
                {"content": {"parts": [{"inlineData": {"data": img_b64, "mimeType": "image/png"}}]}},
            ]
        }
        result = p._extract_gemini_image(data)
        assert result == img_b64


# ==================== OpenAI 图像响应提取测试 ====================

class TestOpenAIImageExtraction:
    def test_extract_from_b64_json(self):
        p = make_provider()
        img_b64 = make_png_base64()
        data = {"data": [{"b64_json": img_b64}]}
        assert p._extract_image_from_openai_response(data) == img_b64

    def test_extract_from_data_url(self):
        p = make_provider()
        img_b64 = make_png_base64()
        data = {"data": [{"url": f"data:image/png;base64,{img_b64}"}]}
        assert p._extract_image_from_openai_response(data) == img_b64

    def test_extract_from_chat_message_multimodal(self):
        p = make_provider()
        img_b64 = make_png_base64()
        data = {
            "choices": [{
                "message": {
                    "content": [
                        {"type": "image", "image_base64": img_b64},
                    ]
                }
            }]
        }
        assert p._extract_image_from_openai_response(data) == img_b64

    def test_extract_http_url(self):
        p = make_provider()
        data = {"data": [{"url": "https://example.com/image.png"}]}
        assert p._extract_http_url_from_openai_response(data) == "https://example.com/image.png"

    def test_extract_http_url_from_choices(self):
        p = make_provider()
        data = {
            "choices": [{
                "message": {
                    "content": "Here is the image: https://cdn.example.com/pic.jpg"
                }
            }]
        }
        url = p._extract_http_url_from_openai_response(data)
        assert url is not None
        assert "cdn.example.com" in url

    def test_extract_from_chat_message_json_string(self):
        p = make_provider()
        img_b64 = make_png_base64()
        json_content = json.dumps({"image_base64": img_b64, "mime_type": "image/png"})
        data = {"choices": [{"message": {"content": json_content}}]}
        assert p._extract_image_from_openai_response(data) == img_b64

    def test_extract_from_message_images_data_url(self):
        p = make_provider()
        img_b64 = make_png_base64()
        data = {
            "choices": [{
                "message": {
                    "content": "text only",
                    "images": [{"image_url": {"url": f"data:image/png;base64,{img_b64}"}}],
                }
            }]
        }
        assert p._extract_image_from_openai_response(data) == img_b64

    def test_extract_from_responses_output(self):
        p = make_provider()
        img_b64 = make_png_base64()
        data = {
            "output": [
                {"type": "reasoning", "content": [{"type": "output_text", "text": "thinking"}]},
                {"type": "image_generation_call", "result": img_b64},
            ]
        }
        assert p._extract_image_from_openai_response(data) == img_b64

    def test_extract_http_url_from_markdown_text(self):
        p = make_provider()
        data = {"choices": [{"message": {"content": "![img](https://img.example.com/a.png)"}}]}
        assert p._extract_http_url_from_openai_response(data) == "https://img.example.com/a.png"

    def test_extract_http_url_from_responses_output(self):
        p = make_provider()
        data = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "image_url", "image_url": {"url": "https://cdn.example.com/r.png"}}],
                }
            ]
        }
        assert p._extract_http_url_from_openai_response(data) == "https://cdn.example.com/r.png"

    def test_no_image_returns_none(self):
        p = make_provider()
        data = {"choices": [{"message": {"content": "Just text, no image"}}]}
        assert p._extract_image_from_openai_response(data) is None


# ==================== 文本生成测试 ====================

class TestTextGeneration:
    def test_text_via_openai(self):
        p = make_provider(text_api_style="openai")
        mock_resp = {"choices": [{"message": {"content": "OpenAI response"}}]}
        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=mock_resp):
            result = run(p.generate_text(
                model_name="gpt-4o",
                contents=[{"type": "text", "text": "Hi"}],
                system_prompt="Be helpful",
                max_attempts=1,
                retry_delay=0,
            ))
        assert result == ["OpenAI response"]

    def test_text_via_gemini(self):
        p = make_provider(text_api_style="gemini")
        mock_resp = {
            "candidates": [{
                "content": {"parts": [{"text": "Gemini response"}]}
            }]
        }
        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=mock_resp):
            result = run(p.generate_text(
                model_name="gemini-2.5-flash",
                contents=[{"type": "text", "text": "Hi"}],
                system_prompt="Be helpful",
                max_attempts=1,
                retry_delay=0,
            ))
        assert result == ["Gemini response"]

    def test_text_retry_on_failure(self):
        p = make_provider(text_api_style="openai")
        mock_resp = {"choices": [{"message": {"content": "Success"}}]}

        call_count = 0
        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("API error")
            return mock_resp

        with patch.object(p, '_post_json', side_effect=mock_post):
            result = run(p.generate_text(
                model_name="test-model",
                contents=[{"type": "text", "text": "Hi"}],
                max_attempts=5,
                retry_delay=0,
            ))
        assert result == ["Success"]
        assert call_count == 3

    def test_text_all_fail(self):
        p = make_provider(text_api_style="openai")
        with patch.object(p, '_post_json', new_callable=AsyncMock, side_effect=Exception("down")):
            result = run(p.generate_text(
                model_name="test",
                contents=[{"type": "text", "text": "Hi"}],
                max_attempts=2,
                retry_delay=0,
            ))
        assert result == ["Error"]

    def test_text_client_error_no_retry(self):
        from providers.multi import ClientError
        p = make_provider(text_api_style="openai")
        with patch.object(p, '_post_json', new_callable=AsyncMock, side_effect=ClientError("HTTP 401")):
            result = run(p.generate_text(
                model_name="test",
                contents=[{"type": "text", "text": "Hi"}],
                max_attempts=5,
                retry_delay=0,
            ))
        assert result == ["Error"]


# ==================== 图像生成测试 ====================

class TestImageGeneration:
    def test_image_via_gemini(self):
        p = make_provider(image_api_style="gemini")
        img_b64 = make_png_base64()
        mock_resp = {
            "candidates": [{
                "content": {
                    "parts": [
                        {"text": "Here's the image"},
                        {"inlineData": {"data": img_b64, "mimeType": "image/png"}},
                    ]
                }
            }]
        }
        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=mock_resp):
            result = run(p.generate_image(
                model_name="gemini-image",
                prompt="A diagram",
                max_attempts=1,
                retry_delay=0,
            ))
        assert len(result) == 1
        assert result[0] == img_b64

    def test_image_via_openai_b64(self):
        p = make_provider(image_api_style="openai")
        img_b64 = make_png_base64()

        chat_resp = {
            "choices": [{
                "message": {
                    "content": [
                        {"type": "image", "image_base64": img_b64},
                    ]
                }
            }]
        }
        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=chat_resp):
            result = run(p.generate_image(
                model_name="dall-e-3",
                prompt="A cat",
                max_attempts=1,
                retry_delay=0,
            ))
        assert len(result) == 1
        assert result[0] == img_b64

    def test_image_via_openai_fallback_to_images_generations(self):
        """chat/completions 没有图片 -> 回退到 images/generations"""
        p = make_provider(image_api_style="openai")
        img_b64 = make_png_base64()

        call_count = 0
        async def mock_post(url, payload, headers, **kwargs):
            nonlocal call_count
            call_count += 1
            if "chat/completions" in url:
                return {"choices": [{"message": {"content": "I can't generate images"}}]}
            else:
                return {"data": [{"b64_json": img_b64}]}

        with patch.object(p, '_post_json', side_effect=mock_post):
            result = run(p.generate_image(
                model_name="dall-e-3",
                prompt="A cat",
                max_attempts=1,
                retry_delay=0,
            ))
        assert call_count == 2
        assert result[0] == img_b64

    def test_image_via_openai_remote_url_download(self):
        """OpenAI 返回远程 URL，需要下载"""
        p = make_provider(image_api_style="openai")
        img_b64 = make_png_base64()

        chat_resp = {"data": [{"url": "https://cdn.example.com/image.png"}]}

        with patch.object(p, '_post_json', new_callable=AsyncMock, return_value=chat_resp), \
             patch.object(p, '_download_image_as_base64', new_callable=AsyncMock, return_value=img_b64):
            result = run(p.generate_image(
                model_name="dall-e-3",
                prompt="A cat",
                max_attempts=1,
                retry_delay=0,
            ))
        assert result[0] == img_b64

    def test_image_all_fail(self):
        p = make_provider(image_api_style="gemini")
        with patch.object(p, '_post_json', new_callable=AsyncMock, side_effect=Exception("API down")):
            result = run(p.generate_image(
                model_name="test",
                prompt="A cat",
                max_attempts=2,
                retry_delay=0,
            ))
        assert result == ["Error"]


# ==================== 混合模式测试 ====================

class TestMixedRouting:
    """测试文本和图像分别路由到不同后端"""

    def test_text_openai_image_gemini(self):
        """text=openai, image=gemini"""
        p = make_provider(text_api_style="openai", image_api_style="gemini")
        img_b64 = make_png_base64()

        text_resp = {"choices": [{"message": {"content": "Text response"}}]}
        image_resp = {
            "candidates": [{
                "content": {"parts": [{"inlineData": {"data": img_b64, "mimeType": "image/png"}}]}
            }]
        }

        captured_urls = []
        async def mock_post(url, payload, headers, **kwargs):
            captured_urls.append(url)
            if "chat/completions" in url:
                return text_resp
            else:
                return image_resp

        async def _run():
            with patch.object(p, '_post_json', side_effect=mock_post):
                text_result = await p.generate_text(
                    model_name="gpt-4o",
                    contents=[{"type": "text", "text": "Hi"}],
                    max_attempts=1, retry_delay=0,
                )
                image_result = await p.generate_image(
                    model_name="gemini-image",
                    prompt="A diagram",
                    max_attempts=1, retry_delay=0,
                )
            return text_result, image_result

        text_result, image_result = run(_run())
        assert text_result == ["Text response"]
        assert image_result[0] == img_b64
        assert any("text-api.example.com" in u for u in captured_urls)
        assert any("image-api.example.com" in u for u in captured_urls)

    def test_text_gemini_image_openai(self):
        """text=gemini, image=openai"""
        p = make_provider(text_api_style="gemini", image_api_style="openai")
        img_b64 = make_png_base64()

        text_resp = {"candidates": [{"content": {"parts": [{"text": "Gemini text"}]}}]}
        image_resp = {"data": [{"b64_json": img_b64}]}

        captured_urls = []
        async def mock_post(url, payload, headers, **kwargs):
            captured_urls.append(url)
            if "generateContent" in url:
                return text_resp
            else:
                return image_resp

        async def _run():
            with patch.object(p, '_post_json', side_effect=mock_post):
                text_result = await p.generate_text(
                    model_name="gemini-2.5-flash",
                    contents=[{"type": "text", "text": "Hi"}],
                    max_attempts=1, retry_delay=0,
                )
                image_result = await p.generate_image(
                    model_name="dall-e-3",
                    prompt="A diagram",
                    max_attempts=1, retry_delay=0,
                )
            return text_result, image_result

        text_result, image_result = run(_run())
        assert text_result == ["Gemini text"]
        assert image_result[0] == img_b64


# ==================== generation_utils 集成测试 ====================

class TestGenerationUtilsMulti:
    def test_init_multi_provider_function_exists(self):
        from utils import generation_utils
        assert hasattr(generation_utils, 'init_multi_provider')
        assert hasattr(generation_utils, 'multi_provider')

    def test_init_multi_provider(self):
        from utils import generation_utils
        generation_utils.init_multi_provider(
            text_api_style="openai",
            text_api_key="text-key",
            text_base_url="https://text.example.com",
            image_api_style="gemini",
            image_api_key="image-key",
            image_base_url="https://image.example.com",
        )
        assert generation_utils.multi_provider is not None
        assert generation_utils.multi_provider.text_api_style == "openai"
        assert generation_utils.multi_provider.image_api_style == "gemini"
        # 清理
        generation_utils.multi_provider = None

    def test_init_multi_provider_skips_empty_key(self):
        from utils import generation_utils
        old = generation_utils.multi_provider
        generation_utils.init_multi_provider(
            text_api_style="openai",
            text_api_key="",
            text_base_url="",
            image_api_style="gemini",
            image_api_key="",
            image_base_url="",
        )
        assert generation_utils.multi_provider is old


# ==================== Agent 路由测试（multi provider） ====================

class TestAgentRoutingMulti:
    @pytest.mark.asyncio
    async def test_planner_routes_multi_to_provider_path(self):
        from utils.config import ExpConfig
        from agents.planner_agent import PlannerAgent
        from utils import generation_utils

        cfg = ExpConfig(
            dataset_name="Test",
            provider="multi",
            task_name="diagram",
            retrieval_setting="none",
            model_name="gemini-2.5-flash",
            image_model_name="gemini-2.0-flash-preview-image-generation",
        )
        agent = PlannerAgent(exp_config=cfg)
        data = {
            "content": "method",
            "visual_intent": "caption",
            "retrieved_examples": [],
        }

        with patch.object(
            generation_utils, "call_evolink_text_with_retry_async",
            new=AsyncMock(return_value=["planner-ok"])
        ) as mock_provider_call, patch.object(
            generation_utils, "call_gemini_with_retry_async",
            new=AsyncMock(return_value=["should-not-be-used"])
        ) as mock_gemini_call:
            result = await agent.process(data)

        assert result["target_diagram_desc0"] == "planner-ok"
        mock_provider_call.assert_awaited_once()
        mock_gemini_call.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_visualizer_routes_multi_to_provider_path(self):
        from utils.config import ExpConfig
        from agents.visualizer_agent import VisualizerAgent
        from utils import generation_utils, image_utils

        cfg = ExpConfig(
            dataset_name="Test",
            provider="multi",
            task_name="diagram",
            retrieval_setting="none",
            model_name="gemini-2.5-flash",
            image_model_name="gemini-2.0-flash-preview-image-generation",
        )
        agent = VisualizerAgent(exp_config=cfg)
        data = {
            "target_diagram_desc0": "draw a diagram",
            "additional_info": {"rounded_ratio": "16:9"},
        }

        with patch.object(
            generation_utils, "call_evolink_image_with_retry_async",
            new=AsyncMock(return_value=[make_png_base64()])
        ) as mock_provider_call, patch.object(
            generation_utils, "call_gemini_with_retry_async",
            new=AsyncMock(return_value=["should-not-be-used"])
        ) as mock_gemini_call, patch.object(
            image_utils, "convert_png_b64_to_jpg_b64",
            return_value="jpg-b64"
        ):
            result = await agent.process(data)

        assert result["target_diagram_desc0_base64_jpg"] == "jpg-b64"
        mock_provider_call.assert_awaited_once()
        mock_gemini_call.assert_not_awaited()


# ==================== ExpConfig Provider 字段测试 ====================

class TestExpConfigMultiProvider:
    def test_config_accepts_multi(self):
        from utils.config import ExpConfig
        config = ExpConfig(dataset_name="Test", provider="multi")
        assert config.provider == "multi"


# ==================== Provider 工厂测试 ====================

class TestProviderFactory:
    def test_create_multi_provider(self):
        from providers import create_provider
        p = create_provider(
            "multi",
            text_api_style="openai",
            text_api_key="k1",
            text_base_url="https://a.com",
            image_api_style="gemini",
            image_api_key="k2",
            image_base_url="https://b.com",
        )
        assert isinstance(p, MultiProvider)
        assert p.text_api_style == "openai"
        assert p.image_api_style == "gemini"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
