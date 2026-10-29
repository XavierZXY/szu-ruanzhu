# -*- coding: utf-8 -*-
"""工具服务端 + 远程客户端的离线联调（Flask test_client 代替真实网络）。"""
import io

import pytest
from PIL import Image

from restoramind.tool_server import create_app
from restoramind.tools.client import RemoteToolClient, ToolServiceLoadBalancer
from restoramind.tools.toolbox import build_toolbox


class _Resp:
    def __init__(self, flask_resp):
        self._r = flask_resp
        self.status_code = flask_resp.status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._r.get_json()


class FlaskSession:
    """把 requests.Session.post 适配到 Flask test_client，便于离线测试。"""

    def __init__(self, app):
        self.client = app.test_client()

    def post(self, url, files=None, data=None, timeout=None):
        payload = dict(data or {})
        for key, (fname, content, mime) in (files or {}).items():
            payload[key] = (io.BytesIO(content), fname, mime)
        path = "/" + url.split("/", 3)[3]
        return _Resp(self.client.post(path, data=payload, content_type="multipart/form-data"))


@pytest.fixture
def app():
    return create_app(family="SCUNet")


def test_health_lists_family_tools(app):
    data = app.test_client().get("/health").get_json()
    assert data["status"] == "ok" and "scunet_real_denoising_gan" in data["tools"]
    assert "dehazeformer_dehaze" not in data["tools"]


def test_process_roundtrip(app, clean_image):
    client = RemoteToolClient(ToolServiceLoadBalancer(["127.0.0.1"]), session=FlaskSession(app))
    out = client.process(clean_image, 5008, "scunet_real_denoising_gan", {})
    assert isinstance(out, Image.Image) and out.size == clean_image.size


def test_process_rejects_unsupported_task(app, clean_image):
    client = RemoteToolClient(ToolServiceLoadBalancer(["127.0.0.1"]), session=FlaskSession(app))
    with pytest.raises(RuntimeError):
        client.process(clean_image, 5008, "dehazeformer_dehaze", {})


def test_remote_toolbox_uses_client_for_deep_tools_and_local_for_classical(clean_image):
    app = create_app()  # 服务全部工具
    client = RemoteToolClient(ToolServiceLoadBalancer(["10.0.0.1", "10.0.0.2"]), session=FlaskSession(app))
    tb = build_toolbox("remote", client=client)
    assert tb.execute("restormer_deraining", clean_image, {}).image.size == clean_image.size
    assert tb.execute("gamma_correction", clean_image, {"gamma": 0.5}).image.size == clean_image.size


def test_load_balancer_round_robin_and_env(monkeypatch):
    lb = ToolServiceLoadBalancer(["a", "b", "c"])
    assert [lb.next_ip() for _ in range(4)] == ["a", "b", "c", "a"]
    monkeypatch.setenv("TOOL_SERVICE_IPS", "1.1.1.1; 2.2.2.2")
    assert ToolServiceLoadBalancer().ips == ["1.1.1.1", "2.2.2.2"]
    monkeypatch.delenv("TOOL_SERVICE_IPS")
    monkeypatch.setenv("TOOL_SERVICE_IP", "9.9.9.9")
    assert ToolServiceLoadBalancer().ips == ["9.9.9.9"]


def test_load_balancer_strategy_from_env(monkeypatch):
    monkeypatch.setenv("TOOL_LOAD_BALANCE_STRATEGY", "random")
    assert ToolServiceLoadBalancer(["a", "b"])._strategy == "random"
    monkeypatch.setenv("TOOL_LOAD_BALANCE_STRATEGY", "weird")
    assert ToolServiceLoadBalancer(["a", "b"])._strategy == "round_robin"
    assert ToolServiceLoadBalancer(["a", "b"], strategy="random")._strategy == "random"
