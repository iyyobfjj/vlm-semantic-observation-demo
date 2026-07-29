from __future__ import annotations

import base64
import os
import time
from pathlib import Path

import httpx
import requests
from dotenv import load_dotenv


def image_to_data_url(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


class VLMClient:
    def __init__(self, model: str | None = None) -> None:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        self.api_style = os.getenv("VLM_API_STYLE", "openai").lower()
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.api_key = os.getenv("API_KEY", "") or dashscope_api_key
        self.base_url = os.getenv("BASE_URL") or (
            "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
            if dashscope_api_key else None
        )
        self.model = model or os.getenv("MODEL_NAME", "") or (
            "qwen3.7-plus" if dashscope_api_key else ""
        )
        self.endpoint = os.getenv("VLM_ENDPOINT", "")
        self.timeout = int(os.getenv("REQUEST_TIMEOUT", "60"))
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "2000"))
        self.force_ipv4 = os.getenv("FORCE_IPV4", "true").lower() not in {
            "0",
            "false",
            "no",
        }
        self.last_call_metrics: dict[str, object] = {}

        if not self.model:
            raise ValueError("MODEL_NAME is required. Copy .env.example to .env and configure it.")

    def analyze_image(self, image_path: Path, system_prompt: str, user_prompt: str) -> str:
        data_url = image_to_data_url(image_path)
        last_error: Exception | None = None
        self.last_call_metrics = {}

        for attempt in range(1, self.max_retries + 1):
            started = time.perf_counter()
            try:
                if self.api_style == "requests":
                    content = self._call_requests_api(data_url, system_prompt, user_prompt)
                else:
                    content = self._call_openai_api(data_url, system_prompt, user_prompt)
                self.last_call_metrics.update(
                    {
                        "attempt": attempt,
                        "elapsed_seconds": round(time.perf_counter() - started, 2),
                    }
                )
                return content
            except Exception as exc:
                last_error = exc
                self.last_call_metrics = {
                    "attempt": attempt,
                    "elapsed_seconds": round(time.perf_counter() - started, 2),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))

        raise RuntimeError(f"VLM request failed after {self.max_retries} attempts: {last_error}")

    def _call_openai_api(self, data_url: str, system_prompt: str, user_prompt: str) -> str:
        from openai import OpenAI

        local_address = "0.0.0.0" if self.force_ipv4 else None
        transport = httpx.HTTPTransport(
            local_address=local_address,
            retries=0,
        )
        http_client = httpx.Client(
            transport=transport,
            timeout=self.timeout,
        )
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=0,
            http_client=http_client,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=self.max_output_tokens,
                extra_body={"enable_thinking": False},
                timeout=self.timeout,
            )
        finally:
            client.close()

        choice = response.choices[0]
        message = choice.message
        reasoning_content = getattr(message, "reasoning_content", None)
        usage = response.usage.model_dump(mode="json") if response.usage else None
        self.last_call_metrics.update(
            {
                "finish_reason": choice.finish_reason,
                "usage": usage,
                "reasoning_content_present": bool(reasoning_content),
                "reasoning_content_chars": len(reasoning_content or ""),
                "network_family": "ipv4" if self.force_ipv4 else "system_default",
            }
        )
        return message.content or ""

    def _call_requests_api(self, data_url: str, system_prompt: str, user_prompt: str) -> str:
        if not self.endpoint:
            raise ValueError("VLM_ENDPOINT is required when VLM_API_STYLE=requests.")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "image": data_url,
            "temperature": 0,
            "max_tokens": self.max_output_tokens,
        }
        response = requests.post(self.endpoint, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            self.last_call_metrics.update(
                {
                    "usage": data.get("usage"),
                    "request_id": data.get("request_id") or data.get("id"),
                }
            )

        if isinstance(data, dict):
            if "text" in data:
                return str(data["text"])
            if "content" in data:
                return str(data["content"])
            choices = data.get("choices")
            if choices:
                message = choices[0].get("message", {})
                return str(message.get("content", ""))
        return str(data)
