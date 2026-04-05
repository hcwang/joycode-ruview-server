import os
import json
import uuid
import secrets
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from csi_processor import get_detector

app = FastAPI(title="JoyCode RuView Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICES_FILE = Path(os.environ.get("DEVICES_FILE", "devices.json"))

def load_devices() -> dict:
    if DEVICES_FILE.exists():
        return json.loads(DEVICES_FILE.read_text())
    return {}

def save_devices(data: dict):
    DEVICES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


# ---- 设备注册 ----

class EnrollRequest(BaseModel):
    mac: str
    name: str
    user_id: str = "default"

class EnrollResponse(BaseModel):
    device_id: str
    token: str

@app.post("/api/devices/enroll", response_model=EnrollResponse)
async def enroll_device(req: EnrollRequest):
    devices = load_devices()
    # 检查是否已注册
    for did, dev in devices.items():
        if dev["mac"] == req.mac and dev["user_id"] == req.user_id:
            return {"device_id": did, "token": dev["token"]}
    # 新注册
    device_id = f"dev-{req.mac.replace(':', '')[:8]}"
    token = secrets.token_hex(16)
    devices[device_id] = {
        "device_id": device_id,
        "mac": req.mac,
        "name": req.name,
        "user_id": req.user_id,
        "token": token,
        "enrolled_at": datetime.now().isoformat(),
    }
    save_devices(devices)
    return {"device_id": device_id, "token": token}

@app.get("/api/devices")
async def list_devices(user_id: str = "default"):
    devices = load_devices()
    return [d for d in devices.values() if d["user_id"] == user_id]

@app.delete("/api/devices/{device_id}")
async def delete_device(device_id: str):
    devices = load_devices()
    if device_id not in devices:
        raise HTTPException(404, "设备不存在")
    del devices[device_id]
    save_devices(devices)
    return {"ok": True}


# ---- WebSocket CSI 接收 ----

active_connections: dict[str, list[WebSocket]] = {}

@app.websocket("/ws/csi/{device_id}")
async def websocket_csi(websocket: WebSocket, device_id: str):
    await websocket.accept()
    if device_id not in active_connections:
        active_connections[device_id] = []
    active_connections[device_id].append(websocket)
    detector = get_detector(device_id)
    try:
        while True:
            data = await websocket.receive_json()
            # 提取 CSI amplitude 均值（简化处理）
            csi_raw = data.get("csi_raw", [])
            if csi_raw:
                amplitude = sum(abs(x) for x in csi_raw) / len(csi_raw)
                detector.add_sample(amplitude)
            # 推理生命体征
            breathing = detector.get_breathing_bpm()
            heart = detector.get_heart_bpm()
            # 构造响应帧
            response = {
                "type": "csi_data",
                "timestamp": data.get("timestamp", 0),
                "device_id": device_id,
                "presence": len(csi_raw) > 0,
                "pose_available": False,
                "vitals": {
                    "breathing_bpm": breathing,
                    "heart_bpm": heart,
                },
                "pose_keypoints": [],
            }
            await websocket.send_json(response)
    except WebSocketDisconnect:
        active_connections[device_id].remove(websocket)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
