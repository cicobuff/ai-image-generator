"""GPU detection and VRAM monitoring using pynvml."""

from dataclasses import dataclass
from typing import Optional
import pynvml

from src.utils.constants import GPU_MEMORY_HEADROOM


@dataclass
class GPUInfo:
    """Information about a single GPU."""
    index: int
    name: str
    total_memory: int  # bytes
    used_memory: int  # bytes
    free_memory: int  # bytes
    utilization: int  # GPU utilization percentage (0-100)
    temperature: int  # GPU temperature in Celsius

    @property
    def total_memory_gb(self) -> float:
        """Total memory in GB."""
        return self.total_memory / (1024 ** 3)

    @property
    def used_memory_gb(self) -> float:
        """Used memory in GB."""
        return self.used_memory / (1024 ** 3)

    @property
    def free_memory_gb(self) -> float:
        """Free memory in GB."""
        return self.free_memory / (1024 ** 3)

    @property
    def usage_percent(self) -> float:
        """Memory usage as percentage."""
        if self.total_memory == 0:
            return 0.0
        return (self.used_memory / self.total_memory) * 100

    @property
    def available_for_model(self) -> float:
        """Memory available for model loading (with headroom) in GB."""
        return max(0, self.free_memory_gb - GPU_MEMORY_HEADROOM)


class GPUManager:
    """Manages GPU detection and VRAM monitoring."""

    def __init__(self):
        self._initialized = False
        self._gpu_count = 0

    def initialize(self) -> bool:
        """Initialize NVML for GPU monitoring."""
        if self._initialized:
            return True

        try:
            pynvml.nvmlInit()
            self._gpu_count = pynvml.nvmlDeviceGetCount()
            self._initialized = True
            return True
        except pynvml.NVMLError as e:
            print(f"Failed to initialize NVML: {e}")
            self._initialized = False
            self._gpu_count = 0
            return False

    def shutdown(self) -> None:
        """Shutdown NVML."""
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass
            self._initialized = False

    @property
    def gpu_count(self) -> int:
        """Get the number of available GPUs."""
        if not self._initialized:
            self.initialize()
        return self._gpu_count

    def get_gpu_info(self, index: int) -> Optional[GPUInfo]:
        """Get information about a specific GPU."""
        if not self._initialized:
            if not self.initialize():
                return None

        if index < 0 or index >= self._gpu_count:
            return None

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

            # Get GPU utilization
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
            except pynvml.NVMLError:
                gpu_util = 0

            # Get GPU temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                temperature = 0

            return GPUInfo(
                index=index,
                name=name,
                total_memory=memory_info.total,
                used_memory=memory_info.used,
                free_memory=memory_info.free,
                utilization=gpu_util,
                temperature=temperature,
            )
        except pynvml.NVMLError as e:
            print(f"Error getting GPU {index} info: {e}")
            return None

    def get_all_gpus(self) -> list[GPUInfo]:
        """Get information about all available GPUs."""
        gpus = []
        for i in range(self.gpu_count):
            info = self.get_gpu_info(i)
            if info:
                gpus.append(info)
        return gpus



# Global GPU manager instance
gpu_manager = GPUManager()
