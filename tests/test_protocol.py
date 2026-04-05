"""
协议验证测试
验证：设备注册 API、WebSocket CSI 数据格式、vitals 字段
"""
import pytest
import json
import asyncio
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import app

client = TestClient(app)

# ---- 设备注册协议 ----

def test_enroll_returns_device_id_and_token():
    resp = client.post("/api/devices/enroll", json={
        "mac": "aa:bb:cc:dd:ee:ff",
        "name": "test-robot",
        "user_id": "test-user"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "device_id" in data, "缺少 device_id 字段"
    assert "token" in data, "缺少 token 字段"
    assert len(data["token"]) > 0

def test_enroll_same_mac_returns_same_device_id():
    resp1 = client.post("/api/devices/enroll", json={
        "mac": "11:22:33:44:55:66", "name": "robot-A", "user_id": "user-1"
    })
    resp2 = client.post("/api/devices/enroll", json={
        "mac": "11:22:33:44:55:66", "name": "robot-A", "user_id": "user-1"
    })
    assert resp1.json()["device_id"] == resp2.json()["device_id"], "相同MAC注册应返回相同device_id"

def test_list_devices():
    client.post("/api/devices/enroll", json={
        "mac": "ff:ee:dd:cc:bb:aa", "name": "robot-list-test", "user_id": "user-list"
    })
    resp = client.get("/api/devices", params={"user_id": "user-list"})
    assert resp.status_code == 200
    devices = resp.json()
    assert isinstance(devices, list)
    assert any(d["mac"] == "ff:ee:dd:cc:bb:aa" for d in devices)

# ---- CSI WebSocket 协议 ----

def test_websocket_csi_response_schema():
    """验证 WebSocket 响应帧包含所有协议字段"""
    with client.websocket_connect("/ws/csi/test-device-001") as ws:
        # 发送 CSI 帧
        ws.send_json({
            "type": "csi_data",
            "timestamp": 1712300000000,
            "device_id": "test-device-001",
            "csi_raw": [10, -5, 8, -3, 12, -7, 9, -4]
        })
        response = ws.receive_json()
        # 验证协议字段
        required_fields = ["type", "timestamp", "device_id", "presence", "pose_available", "vitals", "pose_keypoints"]
        for field in required_fields:
            assert field in response, f"响应缺少必要字段: {field}"
        # 验证字段类型
        assert response["type"] == "csi_data"
        assert isinstance(response["presence"], bool)
        assert isinstance(response["pose_available"], bool)
        assert isinstance(response["pose_keypoints"], list)
        assert "breathing_bpm" in response["vitals"]
        assert "heart_bpm" in response["vitals"]

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
