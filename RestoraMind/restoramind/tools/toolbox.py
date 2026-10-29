# -*- coding: utf-8 -*-
"""
工具箱：智能体的离散动作空间。

Toolbox 统一对外提供 ``execute(tool_name, image, arguments) -> image``，
内部根据工具规格决定走本地经典算子还是远程专家模型服务。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from .classical import run_classical
from .client import RemoteToolClient
from .registry import TOOL_SPECS, ToolSpec, get_tool_spec

Executor = Callable[[ToolSpec, Image.Image, Dict[str, Any]], Image.Image]


class ToolExecutionError(RuntimeError):
    """工具执行失败。"""
    pass


@dataclass
class ToolResult:
    """一次工具执行的结果：图像、工具名、实际参数与耗时。"""
    image: Image.Image
    tool_name: str
    arguments: Dict[str, Any]
    elapsed: float


class Toolbox:
    """智能体的离散动作空间，统一对外提供工具执行。"""
    def __init__(self, executor: Executor, specs: Optional[Dict[str, ToolSpec]] = None):
        self._executor = executor
        self._specs = specs or TOOL_SPECS
        self.call_log: List[ToolResult] = []

    @property
    def tool_names(self) -> List[str]:
        """工具箱中的全部工具名。"""
        return list(self._specs.keys())

    def has(self, name: str) -> bool:
        """工具箱是否包含指定工具。"""
        return name in self._specs

    def spec(self, name: str) -> ToolSpec:
        """获取工具规格，不存在时抛出 ToolExecutionError。"""
        if name not in self._specs:
            raise ToolExecutionError(f"工具箱中不存在工具: {name}")
        return self._specs[name]

    def execute(self, tool_name: str, image: Image.Image, arguments: Optional[Dict[str, Any]] = None) -> ToolResult:
        """规范化参数并执行工具，返回 ToolResult。"""
        spec = self.spec(tool_name)
        params = spec.normalize_arguments(arguments)
        start = time.time()
        try:
            out = self._executor(spec, image, params)
        except Exception as exc:  # noqa: BLE001 - 统一包装为工具执行错误
            raise ToolExecutionError(f"{tool_name} 执行失败: {exc}") from exc
        result = ToolResult(out.convert("RGB"), tool_name, params, time.time() - start)
        self.call_log.append(result)
        return result


def _classical_executor(spec: ToolSpec, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    return Image.fromarray(run_classical(spec.name, arr, params))


def _remote_executor_factory(client: RemoteToolClient) -> Executor:
    def _exec(spec: ToolSpec, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        if spec.is_local:
            return _classical_executor(spec, image, params)
        return client.process(image, spec.port, spec.name, params)
    return _exec


def build_toolbox(mode: str = "local", client: Optional[RemoteToolClient] = None,
                  specs: Optional[Dict[str, ToolSpec]] = None) -> Toolbox:
    """
    mode = "local"  : 全部工具使用本地经典算子（离线 / 测试）
    mode = "remote" : 深度专家模型走远程服务，经典算子本地执行（训练 / 部署）
    """
    if mode == "local":
        return Toolbox(_classical_executor, specs)
    if mode == "remote":
        return Toolbox(_remote_executor_factory(client or RemoteToolClient()), specs)
    raise ValueError(f"未知工具箱模式: {mode}")
