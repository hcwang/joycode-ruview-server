"""
CSI 数据处理模块
- 呼吸率检测：FFT 主频分析，频带 0.1–0.5 Hz（6–30 bpm）
- 心率检测：FFT 主频分析 + 抛物线插值，频带 0.8–2.0 Hz（48–120 bpm）

算法流程：
  1. 去均值 + 去线性趋势（消除直流偏置和慢漂移）
  2. Butterworth bandpass 滤波（抑制带外噪声）
  3. FFT 频谱 + 抛物线插值（提升频率估计精度）
  4. 滑动历史平均（抑制逐帧跳变，平滑输出）
"""
import numpy as np
from scipy.signal import butter, filtfilt, detrend
from collections import deque
from typing import Optional

SAMPLE_RATE = 50          # Hz（时分复用后约 50 Hz）
BUFFER_SIZE = 300         # 样本数，对应 6 秒数据
MIN_SAMPLES = 200         # 至少 4 秒才开始推理
HISTORY_LEN = 5           # 历史输出平均窗口（帧数）


class VitalSignsDetector:
    """
    单设备生命体征检测器，保持 BUFFER_SIZE 个采样点的滑动窗口，
    每次调用 get_*_bpm() 返回最新估计值（已历史平滑）。
    """

    def __init__(self):
        self.buffer: deque[float] = deque(maxlen=BUFFER_SIZE)
        self._br_history: deque[float] = deque(maxlen=HISTORY_LEN)
        self._hr_history: deque[float] = deque(maxlen=HISTORY_LEN)

    def add_sample(self, amplitude: float):
        """追加一个 CSI amplitude 采样点。"""
        self.buffer.append(float(amplitude))

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _bandpass(self, data: np.ndarray, low: float, high: float,
                  fs: float = SAMPLE_RATE, order: int = 4) -> np.ndarray:
        """零相位 Butterworth 带通滤波。"""
        nyq = fs / 2.0
        b, a = butter(order, [low / nyq, high / nyq], btype="band")
        return filtfilt(b, a, data)

    def _fft_dominant_bpm(self, data: np.ndarray, f_low: float, f_high: float,
                          fs: float = SAMPLE_RATE) -> Optional[float]:
        """
        FFT 频谱 + 抛物线插值，返回频带内主频对应的 BPM。
        抛物线插值可将频率分辨率从 fs/N 提升约 3–5 倍。
        """
        n = len(data)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        fft_mag = np.abs(np.fft.rfft(data))

        # 找频带内最大峰
        mask = (freqs >= f_low) & (freqs <= f_high)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None

        sub_mag = fft_mag[indices]
        peak_local = int(np.argmax(sub_mag))
        peak_global = indices[peak_local]

        # 抛物线插值（需要左右邻居）
        if 0 < peak_local < len(sub_mag) - 1:
            alpha = sub_mag[peak_local - 1]
            beta  = sub_mag[peak_local]
            gamma = sub_mag[peak_local + 1]
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-10:
                delta = 0.5 * (alpha - gamma) / denom
                # 限制插值偏移不超过 ±0.5 bin
                delta = np.clip(delta, -0.5, 0.5)
                exact_freq = (peak_global + delta) * fs / n
            else:
                exact_freq = freqs[peak_global]
        else:
            exact_freq = freqs[peak_global]

        # 范围保护
        exact_freq = float(np.clip(exact_freq, f_low, f_high))
        return exact_freq * 60.0  # Hz → BPM

    def _smooth(self, history: deque, new_val: float) -> float:
        """将新值加入历史队列，返回历史均值（平滑输出）。"""
        history.append(new_val)
        return float(np.mean(history))

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def get_breathing_bpm(self) -> Optional[float]:
        """
        估计呼吸率（BPM）。

        - 正常成人呼吸：12–20 bpm（0.2–0.33 Hz）
        - 检测范围：6–30 bpm（0.1–0.5 Hz）
        - 6 秒窗口时频率分辨率约 ±3 bpm（插值后）

        返回 None 表示数据不足（< 4 秒）。
        """
        if len(self.buffer) < MIN_SAMPLES:
            return None

        data = np.array(self.buffer, dtype=np.float64)
        data = detrend(data)                       # 去线性趋势
        data -= np.mean(data)                      # 去直流
        filtered = self._bandpass(data, 0.1, 0.5)
        bpm = self._fft_dominant_bpm(filtered, 0.1, 0.5)
        if bpm is None:
            return None

        smoothed = self._smooth(self._br_history, bpm)
        return round(smoothed, 1)

    def get_heart_bpm(self) -> Optional[float]:
        """
        估计心率（BPM）。

        - 正常成人心率：60–100 bpm（1.0–1.67 Hz）
        - 检测范围：48–120 bpm（0.8–2.0 Hz）
        - 抛物线插值后误差通常 < 1 bpm

        返回 None 表示数据不足（< 4 秒）。
        """
        if len(self.buffer) < MIN_SAMPLES:
            return None

        data = np.array(self.buffer, dtype=np.float64)
        data = detrend(data)
        data -= np.mean(data)
        filtered = self._bandpass(data, 0.8, 2.0)
        bpm = self._fft_dominant_bpm(filtered, 0.8, 2.0)
        if bpm is None:
            return None

        smoothed = self._smooth(self._hr_history, bpm)
        return round(smoothed, 1)


# ------------------------------------------------------------------
# 全局检测器池（按 device_id 隔离）
# ------------------------------------------------------------------

_detectors: dict[str, VitalSignsDetector] = {}


def get_detector(device_id: str) -> VitalSignsDetector:
    """获取或创建指定设备的检测器实例。"""
    if device_id not in _detectors:
        _detectors[device_id] = VitalSignsDetector()
    return _detectors[device_id]
