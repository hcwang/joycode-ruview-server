# joycode-ruview-server

开心小金刚机器人 RuView CSI 数据服务端（FastAPI）

## 功能
- 设备注册与管理（MAC 地址唯一标识，按用户隔离）
- WebSocket 接收 ESP32-S3 上报的 CSI 数据
- 呼吸率/心率轻量推理（bandpass filter）
- 多设备并发数据流处理

## API
- `POST /api/devices/enroll` — 设备注册
- `GET /api/devices` — 列出当前用户设备
- `DELETE /api/devices/{device_id}` — 解绑设备
- `WS /ws/csi/{device_id}` — CSI 数据推送

## 运行
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8080
```
