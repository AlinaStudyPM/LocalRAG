# benchmarks/benchmark_utils.py
import os
import time
import tracemalloc
from typing import Callable, Any, Tuple
import threading

import psutil


def benchmark_call(func: Callable, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    Оборачивает вызов func, измеряя время выполнения и потребление памяти.

    Returns:
        (result, latency_sec, peak_memory_mb)
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        result = func(*args, **kwargs)
    finally:
        t1 = time.perf_counter()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    latency = t1 - t0
    peak_memory_mb = peak / (1024 * 1024)
    return result, latency, peak_memory_mb

def benchmark_call_psutil(func: Callable, interval_sec: float = 0.05, *args, **kwargs) -> Tuple[Any, float, float]:
    """
    Оборачивает вызов func, измеряя время выполнения и пиковое потребление
    памяти текущего процесса Python через psutil с периодическим опросом.

    Returns:
        (result, latency_sec, peak_memory_mb)
    """
    process = psutil.Process(os.getpid())
    peak_memory_mb = 0.0
    result = None
    finished = False

    def _poll_memory():
        nonlocal peak_memory_mb, finished
        while not finished:
            try:
                mem = process.memory_info().rss / (1024 * 1024)
                if mem > peak_memory_mb:
                    peak_memory_mb = mem
            except psutil.NoSuchProcess:
                break
            time.sleep(interval_sec)

    t0 = time.perf_counter()
    poll_thread = threading.Thread(target=_poll_memory)
    poll_thread.start()

    try:
        result = func(*args, **kwargs)
    finally:
        finished = True
        poll_thread.join()
        t1 = time.perf_counter()

    latency = t1 - t0
    return result, latency, peak_memory_mb

if __name__ == "__main__":
    def dummy_work(duration_sec=0.1, alloc_mb=5):
        time.sleep(duration_sec)
        data = bytearray(int(alloc_mb * 1024 * 1024))
        return len(data)

    print("Тест benchmark_call:")
    result, latency, memory = benchmark_call(dummy_work, 0.15, 2.0)

    assert result == int(2.0 * 1024 * 1024), f"Неверный результат: {result}"
    assert latency >= 0.15, f"Latency слишком мал: {latency:.3f} сек"
    assert memory >= 1.0, f"Memory слишком мала: {memory:.1f} МБ"

    print(f"  Результат функции : {result} байт")
    print(f"  Latency           : {latency:.3f} сек (ожидалось >= 0.15)")
    print(f"  Peak memory       : {memory:.1f} МБ (ожидалось >= 1.0)")
    print("✅ Тест пройден.")

    print("Тест benchmark_call_psutil:")
    result3, latency3, memory3 = benchmark_call_psutil(dummy_work, 0.05, 0.15, 2.0)
    print(f"  Результат функции : {result3} байт")
    print(f"  Latency           : {latency3:.3f} сек")
    print(f"  Peak memory       : {memory3:.1f} МБ")
    print("✅ Тест пройден.")
