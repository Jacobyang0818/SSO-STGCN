import pynvml
import time


def monitor_gpu_usage(process, max_usage_percent, stop_event, gpu_id=0):
    """監控 GPU 記憶體使用百分比，若超過閾值則終止訓練進程。"""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory_mb = mem_info.total / 1024**2  # 總記憶體 (MB)
        max_allowed_mb = total_memory_mb * max_usage_percent  # 允許的最大 MB
        # print(f"[監控] GPU 總記憶體: {total_memory_mb:.2f} MB，允許最大使用: {max_allowed_mb:.2f} MB ({max_usage_percent*100}%)")
    except pynvml.NVMLError as e:
        print(f"🚨 GPU 監控初始化失敗: {e}")
        stop_event.set()
        return

    while process.is_alive() and not stop_event.is_set():
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used_memory_mb = mem_info.used / 1024**2
        usage_percent = used_memory_mb / total_memory_mb  # 當前使用百分比
        # print(f"[監控] 目前 GPU 記憶體使用量: {used_memory_mb:.2f} MB ({usage_percent*100:.2f}%)")
        if usage_percent > max_usage_percent:
            print(f"🚨 GPU 記憶體使用超過 {max_usage_percent*100}%，終止訓練!")
            stop_event.set()
            break
        time.sleep(1)
    pynvml.nvmlShutdown()
