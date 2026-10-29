# -*- coding: utf-8 -*-
"""
复原工具 HTTP 服务（Flask）。

每个专家模型家族启动一个服务进程，对外暴露：
    GET  /health            健康检查，返回支持的工具列表
    POST /process           multipart image + form(task, 参数...) → {"success", "image"(base64 PNG)}
"""
from __future__ import annotations

import base64
import io
from typing import Dict, Iterable, Optional

from flask import Flask, jsonify, request
from PIL import Image

from ..tools.registry import TOOL_SPECS, get_tool_spec
from .backends import Backend, ClassicalBackend


def create_app(backend: Optional[Backend] = None, family: Optional[str] = None,
               tools: Optional[Iterable[str]] = None) -> Flask:
    """
    backend : 处理后端，默认经典算子
    family  : 仅服务该模型家族的工具（与端口一一对应）
    tools   : 显式指定服务的工具名集合（优先于 family）
    """
    backend = backend or ClassicalBackend()
    if tools is not None:
        served = {n for n in tools}
    elif family is not None:
        served = {n for n, s in TOOL_SPECS.items() if s.family == family}
    else:
        served = set(TOOL_SPECS.keys())

    app = Flask("restoramind_tool_server")
    app.config["SERVED_TOOLS"] = sorted(served)

    @app.get("/health")
    def health():
        """健康检查，返回服务支持的工具列表。"""
        return jsonify({"status": "ok", "tools": app.config["SERVED_TOOLS"]})

    @app.post("/process")
    def process():
        """处理一次复原请求并返回 base64 编码的结果图。"""
        task = request.form.get("task", "")
        if task not in served:
            return jsonify({"success": False, "error": f"unsupported task '{task}'"}), 400
        if "image" not in request.files:
            return jsonify({"success": False, "error": "missing image"}), 400
        try:
            image = Image.open(io.BytesIO(request.files["image"].read())).convert("RGB")
            spec = get_tool_spec(task)
            params: Dict[str, str] = {k: v for k, v in request.form.items() if k != "task"}
            out = backend.process(task, image, spec.normalize_arguments(params))
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            return jsonify({"success": True, "image": base64.b64encode(buf.getvalue()).decode("ascii"),
                            "width": out.width, "height": out.height})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"success": False, "error": str(exc)}), 500

    return app
