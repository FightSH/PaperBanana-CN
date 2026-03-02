# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Visualizer Agent - 将详细描述转换为图像或代码。
"""

from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Any
import base64, io, asyncio, re
import matplotlib.pyplot as plt
from PIL import Image

from utils import generation_utils, image_utils
from .base_agent import BaseAgent


def _execute_plot_code_worker(code_text: str, log_prefix: str = "") -> str:
    """
    Independent plot code execution worker:
    1. Extract code
    2. Execute plotting
    3. Return JPEG as Base64 string
    """
    match = re.search(r"```python(.*?)```", code_text, re.DOTALL)
    code_clean = match.group(1).strip() if match else code_text.strip()

    plt.switch_backend("Agg")
    plt.close("all")
    plt.rcdefaults()

    try:
        exec_globals = {}
        exec(code_clean, exec_globals)
        if plt.get_fignums():
            buf = io.BytesIO()
            plt.savefig(buf, format="jpeg", bbox_inches="tight", dpi=300)
            plt.close("all")

            buf.seek(0)
            img_bytes = buf.read()
            return base64.b64encode(img_bytes).decode("utf-8")
        else:
            return None

    except Exception as e:
        prefix = f"{log_prefix} " if log_prefix else ""
        print(f"Error executing plot code {prefix}: {e}")
        return None


class VisualizerAgent(BaseAgent):
    """Visualizer Agent to generate images based on user queries"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Task-specific configurations
        if "plot" in self.exp_config.task_name:
            self.model_name = self.exp_config.model_name
            self.system_prompt = PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT
            self.process_executor = ProcessPoolExecutor(max_workers=32)
            self.task_config = {
                "task_name": "plot",
                "use_image_generation": False,
                "prompt_template": "Use python matplotlib to generate a statistical plot based on the following detailed description: {desc}\n Only provide the code without any explanations. Code:",
                "max_output_tokens": 50000,
            }
        else:
            self.model_name = self.exp_config.image_model_name
            self.system_prompt = DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT
            self.process_executor = None
            self.task_config = {
                "task_name": "diagram",
                "use_image_generation": True,
                "prompt_template": "Render an image based on the following detailed description: {desc}\n Note that do not include figure titles in the image. Diagram: ",
                "max_output_tokens": 50000,
            }

    def __del__(self):
        if self.process_executor:
            self.process_executor.shutdown(wait=True)

    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.task_config
        task_name = cfg["task_name"]
        candidate_id = data.get("candidate_id", "N/A")
        log_prefix = f"[candidate={candidate_id}]"
        print(
            f"[DEBUG] [VisualizerAgent] {log_prefix} 开始处理, task={task_name}, "
            f"provider={self.exp_config.provider}, model={self.model_name}, 图像生成={cfg['use_image_generation']}"
        )

        desc_keys_to_process = []
        for key in [
            f"target_{task_name}_desc0",
            f"target_{task_name}_stylist_desc0",
        ]:
            if key in data and f"{key}_base64_jpg" not in data:
                desc_keys_to_process.append(key)

        for round_idx in range(3):
            key = f"target_{task_name}_critic_desc{round_idx}"
            if key in data and f"{key}_base64_jpg" not in data:
                critic_suggestions_key = f"target_{task_name}_critic_suggestions{round_idx}"
                critic_suggestions = data.get(critic_suggestions_key, "")

                if critic_suggestions.strip() == "No changes needed." and round_idx > 0:
                    prev_base64_key = f"target_{task_name}_critic_desc{round_idx - 1}_base64_jpg"
                    if prev_base64_key in data:
                        data[f"{key}_base64_jpg"] = data[prev_base64_key]
                        print(
                            f"[Visualizer] {log_prefix} Reused base64 from round {round_idx - 1} for {key}"
                        )
                        continue

                desc_keys_to_process.append(key)

        if not cfg["use_image_generation"]:
            loop = asyncio.get_running_loop()

        print(f"[DEBUG] [VisualizerAgent] {log_prefix} 待处理 desc_keys: {desc_keys_to_process}")
        image_max_attempts = self.exp_config.image_max_attempts
        image_retry_delay = self.exp_config.image_retry_delay
        image_poll_interval = self.exp_config.image_poll_interval
        text_max_attempts = 5
        text_retry_delay = 30

        for desc_key in desc_keys_to_process:
            prompt_text = cfg["prompt_template"].format(desc=data[desc_key])
            content_list = [{"type": "text", "text": prompt_text}]
            step_context = f"candidate={candidate_id}, desc_key={desc_key}"
            print(
                f"[DEBUG] [VisualizerAgent] {log_prefix} 处理 {desc_key}, prompt 长度={len(prompt_text)}"
            )

            # 根据 provider 路由 API 调用
            if self.exp_config.provider in ("evolink", "multi"):
                if cfg["use_image_generation"]:
                    # Evolink/Multi 图像生成
                    aspect_ratio = "1:1"
                    if "additional_info" in data and "rounded_ratio" in data["additional_info"]:
                        aspect_ratio = data["additional_info"]["rounded_ratio"]

                    response_list = await generation_utils.call_evolink_image_with_retry_async(
                        model_name=self.model_name,
                        prompt=prompt_text,
                        config={
                            "aspect_ratio": aspect_ratio,
                            "quality": "2K",
                            "poll_interval": image_poll_interval,
                        },
                        max_attempts=image_max_attempts,
                        retry_delay=image_retry_delay,
                        error_context=step_context,
                    )
                else:
                    # Evolink/Multi 文本生成（用于代码生成）
                    response_list = await generation_utils.call_evolink_text_with_retry_async(
                        model_name=self.exp_config.model_name,
                        contents=content_list,
                        config={
                            "system_prompt": self.system_prompt,
                            "temperature": self.exp_config.temperature,
                            "max_output_tokens": cfg["max_output_tokens"],
                        },
                        max_attempts=text_max_attempts,
                        retry_delay=text_retry_delay,
                        error_context=step_context,
                    )
            elif "gemini" in self.model_name:
                from google.genai import types
                gen_config_args = {
                    "system_instruction": self.system_prompt,
                    "temperature": self.exp_config.temperature,
                    "candidate_count": 1,
                    "max_output_tokens": cfg["max_output_tokens"],
                }
                if cfg["use_image_generation"]:
                    aspect_ratio = "1:1"
                    if "additional_info" in data and "rounded_ratio" in data["additional_info"]:
                        aspect_ratio = data["additional_info"]["rounded_ratio"]
                    gen_config_args["response_modalities"] = ["IMAGE"]
                    gen_config_args["image_config"] = types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size="1k",
                    )
                response_list = await generation_utils.call_gemini_with_retry_async(
                    model_name=self.model_name,
                    contents=content_list,
                    config=types.GenerateContentConfig(**gen_config_args),
                    max_attempts=image_max_attempts if cfg["use_image_generation"] else text_max_attempts,
                    retry_delay=image_retry_delay if cfg["use_image_generation"] else text_retry_delay,
                    error_context=step_context,
                )
            elif "gpt-image" in self.model_name:
                image_config = {
                    "size": "1536x1024",
                    "quality": "high",
                    "background": "opaque",
                    "output_format": "png",
                }
                response_list = await generation_utils.call_openai_image_generation_with_retry_async(
                    model_name=self.model_name,
                    prompt=prompt_text,
                    config=image_config,
                    max_attempts=image_max_attempts,
                    retry_delay=image_retry_delay,
                    error_context=step_context,
                )
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

            if not response_list or not response_list[0]:
                raise RuntimeError(f"[VisualizerAgent] {log_prefix} {desc_key}: API 返回空响应")

            print(
                f"[DEBUG] [VisualizerAgent] {log_prefix} {desc_key}: API 响应长度={len(response_list[0])}, "
                f"值前20字={response_list[0][:20]}..."
            )

            # Post-process based on task type
            if cfg["use_image_generation"]:
                converted_jpg = await asyncio.to_thread(
                    image_utils.convert_png_b64_to_jpg_b64, response_list[0]
                )
                if converted_jpg:
                    data[f"{desc_key}_base64_jpg"] = converted_jpg
                    print(
                        f"[DEBUG] [VisualizerAgent] {log_prefix} ✓ {desc_key}_base64_jpg 已生成, "
                        f"大小={len(converted_jpg)}"
                    )
                else:
                    raise RuntimeError(f"[VisualizerAgent] {log_prefix} {desc_key}: 图像转换失败")
            else:
                raw_code = response_list[0]

                if not hasattr(self, "process_executor") or self.process_executor is None:
                    self.process_executor = ProcessPoolExecutor(max_workers=4)

                base64_jpg = await loop.run_in_executor(
                    self.process_executor, _execute_plot_code_worker, raw_code, log_prefix
                )
                data[f"{desc_key}_code"] = raw_code

                if base64_jpg:
                    data[f"{desc_key}_base64_jpg"] = base64_jpg

        return data


DIAGRAM_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert scientific diagram illustrator. Generate high-quality scientific diagrams based on user requests."""

PLOT_VISUALIZER_AGENT_SYSTEM_PROMPT = """You are an expert statistical plot illustrator. Write code to generate high-quality statistical plots based on user requests."""
