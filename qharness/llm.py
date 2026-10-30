"""LLM 接口。

每次调用传 role（哪个节点在调）、messages（提示词）、context（拼提示词用的结构化输入，留作日志和扩展）。
- AnthropicLLM：Anthropic Messages 接口（含 DeepSeek 等兼容端点）。
- OpenAICompatLLM：OpenAI 兼容接口。
  默认关闭推理模型的思考（QH_LLM_THINKING=default 恢复服务默认）：这里都是短的结构化任务，
  开着思考会把 max_tokens 先花在思考上、延迟高。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request


def extract_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    raw = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(raw)


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    return (m.group(1) if m else text).strip() + "\n"


class LLM:
    name = "base"
    supports_tools = False  # 原生 tool calling；不支持的接口由调用方退回文本模式

    def __init__(self):
        self.calls: list[dict] = []  # 每次调用的角色、耗时、token、出错信息
        self.calls_file = None       # 设了就每次调用当场追加一行（进程被杀也不丢）

    def chat(self, role: str, messages: list[dict], context: dict) -> str:
        raise NotImplementedError

    def respond(self, role: str, system: str, messages: list[dict], tools: list[dict], context: dict) -> list[dict]:
        """带工具的一轮对话，返回原始内容块（text / tool_use）。messages 由调用方保证 user/assistant 交替。"""
        raise NotImplementedError


def thinking_disabled() -> bool:
    return os.environ.get("QH_LLM_THINKING", "off").strip().lower() != "default"


def load_env_file(path: str, prefixes=("ANTHROPIC_", "QH_LLM_")) -> list[str]:
    """从 .env 读取模型配置到本进程环境变量，只取指定前缀，返回读到的变量名（不返回值）。"""
    names = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k.startswith(prefixes) and v:
            os.environ[k] = v
            names.append(k)
    return names


class _HTTPLLM(LLM):
    retries = 2
    _record_lock = threading.Lock()

    def _record(self, call: dict):
        with self._record_lock:
            self.calls.append(call)
            if self.calls_file:
                with open(self.calls_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(call, ensure_ascii=False) + "\n")

    def _post(self, url: str, body: dict, headers: dict, role: str) -> dict:
        data = json.dumps(body).encode()
        last = None
        for attempt in range(self.retries + 1):
            t = time.time()
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                self._record({"role": role, "latency_s": round(time.time() - t, 2), "usage": out.get("usage"),
                              "attempt": attempt})
                return out
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
                if e.code < 500 and e.code != 429:
                    break
            except Exception as e:  # noqa: BLE001  超时、断连都重试
                last = f"{type(e).__name__}: {e}"
            time.sleep(2 * (attempt + 1))
        self._record({"role": role, "error": last})
        raise RuntimeError(f"LLM 调用失败（{role}）：{last}")


class OpenAICompatLLM(_HTTPLLM):
    """环境变量：QH_LLM_BASE_URL（如 https://api.deepseek.com/v1）、QH_LLM_API_KEY、QH_LLM_MODEL。"""
    name = "openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None, model: str | None = None,
                 temperature: float = 0.2, timeout: int = 120):
        super().__init__()
        self.base_url = (base_url or os.environ.get("QH_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("QH_LLM_API_KEY", "")
        self.model = model or os.environ.get("QH_LLM_MODEL", "")
        if not (self.base_url and self.api_key and self.model):
            raise RuntimeError("需要设置 QH_LLM_BASE_URL / QH_LLM_API_KEY / QH_LLM_MODEL")
        self.temperature, self.timeout = temperature, timeout

    def chat(self, role, messages, context):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if thinking_disabled():
            body["thinking"] = {"type": "disabled"}
        out = self._post(f"{self.base_url}/chat/completions", body, {"Authorization": f"Bearer {self.api_key}"}, role)
        return out["choices"][0]["message"]["content"]


class AnthropicLLM(_HTTPLLM):
    """Anthropic Messages 接口（含 DeepSeek 等兼容端点）。环境变量：ANTHROPIC_API_KEY、ANTHROPIC_BASE_URL、ANTHROPIC_MODEL。"""
    name = "anthropic"
    supports_tools = True

    def __init__(self, max_tokens: int = 4096, temperature: float = 0.2, timeout: int = 180):
        super().__init__()
        self.base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = os.environ.get("ANTHROPIC_MODEL", "")
        if not (self.api_key and self.model):
            raise RuntimeError("需要设置 ANTHROPIC_API_KEY / ANTHROPIC_MODEL（可用 --env-file 读取）")
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout

    def chat(self, role, messages, context):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        msgs = [m for m in messages if m["role"] != "system"]
        # Messages 接口要求 user/assistant 交替：相邻同角色合并
        merged: list[dict] = []
        for m in msgs:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1] = {"role": m["role"], "content": merged[-1]["content"] + "\n\n" + m["content"]}
            else:
                merged.append(dict(m))
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "（上文）"})
        body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
                "system": system, "messages": merged}
        if thinking_disabled():
            body["thinking"] = {"type": "disabled"}
        out = self._post(f"{self.base_url}/v1/messages", body,
                         {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, role)
        return "".join(b.get("text", "") for b in out.get("content", []) if b.get("type") == "text")


def _anthropic_respond(self, role, system, messages, tools, context):
    body = {"model": self.model, "max_tokens": self.max_tokens, "temperature": self.temperature,
            "system": system, "messages": messages, "tools": tools}
    if thinking_disabled():
        body["thinking"] = {"type": "disabled"}
    out = self._post(f"{self.base_url}/v1/messages", body,
                     {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}, role)
    return [b for b in out.get("content", []) if b.get("type") in ("text", "tool_use")]


AnthropicLLM.respond = _anthropic_respond


def make_llm(kind: str = "anthropic", env_file: str | None = None) -> LLM:
    env_file = env_file or os.environ.get("QH_ENV_FILE")
    if env_file:
        load_env_file(env_file)
    return {"anthropic": AnthropicLLM, "openai": OpenAICompatLLM}[kind]()
