# -*- coding: utf-8 -*-
"""
远程复原工具服务客户端。

协议（与工具服务端 restoramind.tool_server 一致）：
    POST http://{ip}:{port}/process
        multipart: image=<PNG 字节>
        form:      task=<工具名>, 其余为工具参数
    响应 JSON: {"success": true, "image": "<base64 PNG>"} 或 {"success": false, "error": "..."}
支持多服务器轮询/随机负载均衡（环境变量 TOOL_SERVICE_IPS 配置地址，逗号分隔；TOOL_LOAD_BALANCE_STRATEGY 选择策略）。
"""
from __future__ import annotations

import base64
import io
import os
import random
import threading
from typing import Any, Dict, List, Optional

import requests
from PIL import Image


class ToolServiceLoadBalancer:
    """多 IP 负载均衡器。"""

    def __init__(self, ips: Optional[List[str]] = None, strategy: Optional[str] = None):
        if ips is None:
            raw = os.environ.get("TOOL_SERVICE_IPS", "").strip()
            if raw:
                ips = [x.strip() for x in raw.replace(";", ",").split(",") if x.strip()]
            else:
                ips = [os.environ.get("TOOL_SERVICE_IP", "127.0.0.1").strip()]
        if not ips:
            raise ValueError("至少需要一个工具服务 IP")
        self._ips = list(ips)
        # 策略优先级：构造参数 > 环境变量 TOOL_LOAD_BALANCE_STRATEGY > round_robin
        strategy = strategy or os.environ.get("TOOL_LOAD_BALANCE_STRATEGY", "round_robin").strip().lower()
        self._strategy = strategy if strategy in ("round_robin", "random") else "round_robin"
        self._index = 0
        self._lock = threading.Lock()

    @property
    def ips(self) -> List[str]:
        """已配置的服务器地址列表。"""
        return list(self._ips)

    def next_ip(self) -> str:
        """按策略返回下一个服务器地址。"""
        if len(self._ips) == 1:
            return self._ips[0]
        if self._strategy == "random":
            return random.choice(self._ips)
        with self._lock:
            ip = self._ips[self._index % len(self._ips)]
            self._index += 1
        return ip


def encode_image(image: Image.Image) -> bytes:
    """把图像编码为 PNG 字节。"""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def decode_image_b64(data: str) -> Image.Image:
    """解码 base64 编码的 PNG 图像。"""
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


class RemoteToolClient:
    """通过 HTTP 调用远程专家模型服务。"""

    def __init__(self, balancer: Optional[ToolServiceLoadBalancer] = None, timeout: float = 300.0,
                 session: Optional[Any] = None):
        self.balancer = balancer or ToolServiceLoadBalancer()
        self.timeout = timeout
        # session 可注入（例如 Flask test_client 适配器）以便离线测试
        self.session = session or requests.Session()

    def url_for(self, port: int) -> str:
        """构造指定端口的服务请求地址（含负载均衡）。"""
        return f"http://{self.balancer.next_ip()}:{port}/process"

    def process(self, image: Image.Image, port: int, task: str, params: Dict[str, Any]) -> Image.Image:
        """向远程服务提交图像并返回复原结果。"""
        files = {"image": ("image.png", encode_image(image), "image/png")}
        data = {"task": task, **{k: str(v) for k, v in params.items()}}
        resp = self.session.post(self.url_for(port), files=files, data=data, timeout=self.timeout)
        resp.raise_for_status()
        result = resp.json()
        if not result.get("success"):
            raise RuntimeError(f"工具服务返回错误: {result.get('error', 'unknown')}")
        return decode_image_b64(result["image"])
