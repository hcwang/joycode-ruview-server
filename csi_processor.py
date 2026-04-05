"""
CSI 数据处理模块
- 呼吸率检测：bandpass 0.1-0.5 Hz
- 心率检测：bandpass 0.8-2.0 Hz
"""
import numpy as np
from scipy.signal import butter, filtfilt
from collections import deque
from typing import Optional

SAMPLE_RATE = 50  # Hz (时分复用后约50Hz)
BUFFER_SIZE = 300  # 6秒数据

class VitalSignsDetector:
    def __init__(self):
        self.buffer = deque(maxlen=BUFFER_SIZE)

    def add_sample(self, amplitude: float):
        self.buffer.append(amplitude)

    def _bandpass(self, data, low, high, fs=SAMPLE_RATE, order=4):
        nyq = fs / 2
        b, a = butter(order, [low/nyq, high/nyq], btype='band')
        return filtfilt(b, a, data)

    def _count_peaks(self, signal, min_interval_samples):
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                if not peaks or (i - peaks[-1]) >= min_interval_samples:
                    peaks.append(i)
        return len(peaks)

    def get_breathing_bpm(self) -> Optional[float]:
        if len(self.buffer) < SAMPLE_RATE * 4:
            return None
        data = np.array(self.buffer)
        filtered = self._bandpass(data, 0.1, 0.5)
        peaks = self._count_peaks(filtered, SAMPLE_RATE * 2)  # 最小间隔2秒
        duration_min = len(data) / SAMPLE_RATE / 60
        return round(peaks / duration_min, 1)

    def get_heart_bpm(self) -> Optional[float]:
        if len(self.buffer) < SAMPLE_RATE * 4:
            return None
        data = np.array(self.buffer)
        filtered = self._bandpass(data, 0.8, 2.0)
        peaks = self._count_peaks(filtered, int(SAMPLE_RATE * 0.4))  # 最小间隔0.4秒
        duration_min = len(data) / SAMPLE_RATE / 60
        return round(peaks / duration_min, 1)


# 全局检测器池（按 device_id 隔离）
_detectors: dict[str, VitalSignsDetector] = {}

def get_detector(device_id: str) -> VitalSignsDetector:
    if device_id not in _detectors:
        _detectors[device_id] = VitalSignsDetector()
    return _detectors[device_id]
