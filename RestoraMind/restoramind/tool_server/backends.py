# -*- coding: utf-8 -*-
"""
工具服务端后端。

Backend 把「工具名 + 图像 + 参数」映射为复原图像。
    ClassicalBackend : 经典算子实现，开箱即用
    CallableBackend  : 以 ``module:function`` 形式加载任意用户实现（例如封装
                       SwinIR / Restormer / SCUNet 等预训练模型的推理函数），
                       函数签名须为 fn(tool_name: str, image: PIL.Image, params: dict) -> PIL.Image
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np
from PIL import Image

from ..tools.classical import run_classical


class Backend(ABC):
    """工具服务后端抽象：把（工具名，图像，参数）映射为复原图像。"""
    @abstractmethod
    def process(self, tool_name: str, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """执行指定工具并返回复原图像。"""
        ...

    def supports(self, tool_name: str) -> bool:  # noqa: D401
        """后端是否支持该工具。"""
        return True


class ClassicalBackend(Backend):
    """经典算子后端，开箱即用。"""
    def process(self, tool_name: str, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """使用本地经典算子执行工具。"""
        return Image.fromarray(run_classical(tool_name, np.array(image.convert("RGB")), params))


class CallableBackend(Backend):
    """把任意可调用对象包装为后端，用于接入预训练模型。"""
    def __init__(self, fn: Callable[[str, Image.Image, Dict[str, Any]], Image.Image]):
        self.fn = fn

    def process(self, tool_name: str, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """调用用户函数执行工具。"""
        return self.fn(tool_name, image, params).convert("RGB")


def load_backend(spec: str) -> Backend:
    """``classical`` 或 ``package.module:function``。"""
    if spec == "classical":
        return ClassicalBackend()
    if ":" not in spec:
        raise ValueError("自定义后端需使用 module:function 形式")
    module_name, fn_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return CallableBackend(getattr(module, fn_name))
