"""
Hardware & System Resource Profiler
Calculates real-time GPU and CPU hardware metrics, VRAM allocation, 
and measures live PyTorch throughput (Images/sec, latency, memory peak).
"""

import os
import sys
import time
import platform
import subprocess
import torch
import numpy as np

def get_cpu_and_ram_info():
    info = {}
    info["os"] = f"{platform.system()} {platform.release()} ({platform.version()})"
    info["processor"] = platform.processor() or "x86_64"
    info["cpu_cores_logical"] = os.cpu_count() or 1
    
    # Try using psutil if available, otherwise fallback to OS APIs
    try:
        import psutil
        info["cpu_usage_pct"] = psutil.cpu_percent(interval=0.5)
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total / (1024 ** 3)
        info["ram_used_gb"] = vm.used / (1024 ** 3)
        info["ram_free_gb"] = vm.available / (1024 ** 3)
        info["ram_usage_pct"] = vm.percent
    except ImportError:
        info["cpu_usage_pct"] = "Install psutil for real-time CPU %"
        info["ram_total_gb"] = "N/A"
        info["ram_used_gb"] = "N/A"
        info["ram_free_gb"] = "N/A"
        info["ram_usage_pct"] = "N/A"
        
    return info

def get_gpu_info():
    info = {}
    info["cuda_available"] = torch.cuda.is_available()
    
    if not info["cuda_available"]:
        info["status"] = "CUDA is not available. Running on CPU."
        return info
        
    info["device_count"] = torch.cuda.device_count()
    info["current_device_id"] = torch.cuda.current_device()
    info["gpu_name"] = torch.cuda.get_device_name(info["current_device_id"])
    info["cuda_version"] = torch.version.cuda
    info["cudnn_version"] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
    
    # VRAM stats in MB and GB
    total_mem = torch.cuda.get_device_properties(info["current_device_id"]).total_memory
    allocated_mem = torch.cuda.memory_allocated(info["current_device_id"])
    reserved_mem = torch.cuda.memory_reserved(info["current_device_id"])
    max_allocated = torch.cuda.max_memory_allocated(info["current_device_id"])
    
    info["vram_total_mb"] = total_mem / (1024 ** 2)
    info["vram_total_gb"] = total_mem / (1024 ** 3)
    info["vram_allocated_mb"] = allocated_mem / (1024 ** 2)
    info["vram_reserved_mb"] = reserved_mem / (1024 ** 2)
    info["vram_max_allocated_mb"] = max_allocated / (1024 ** 2)
    info["vram_free_mb"] = (total_mem - reserved_mem) / (1024 ** 2)
    
    # Try querying nvidia-smi for temperature and power if possible
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
            encoding="utf-8"
        ).strip()
        parts = [p.strip() for p in smi_out.split(",")]
        if len(parts) >= 3:
            info["gpu_temp_c"] = parts[0]
            info["gpu_util_pct"] = parts[1]
            info["gpu_power_w"] = parts[2]
    except Exception:
        info["gpu_temp_c"] = "N/A"
        info["gpu_util_pct"] = "N/A"
        info["gpu_power_w"] = "N/A"
        
    return info

def run_live_gpu_benchmark(batch_size=32, num_batches=30):
    """
    Executes a live throughput benchmark using 3-Channel 256x256 ResNet forward-backward passes.
    """
    if not torch.cuda.is_available():
        return None
        
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Build a lightweight benchmark model (ResNet-18)
    from model import build_model
    model = build_model(arch="resnet18", num_classes=2, pretrained=False, in_channels=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Warmup
    dummy_input = torch.randn(batch_size, 3, 256, 256, device=device)
    dummy_target = torch.randint(0, 2, (batch_size,), device=device)
    for _ in range(5):
        optimizer.zero_grad()
        out = model(dummy_input)
        loss = criterion(out, dummy_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    
    # Timed Benchmark
    start_time = time.time()
    for _ in range(num_batches):
        optimizer.zero_grad()
        out = model(dummy_input)
        loss = criterion(out, dummy_target)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    
    total_images = batch_size * num_batches
    throughput_fps = total_images / elapsed_time
    latency_per_batch_ms = (elapsed_time / num_batches) * 1000
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    return {
        "batch_size": batch_size,
        "num_batches": num_batches,
        "total_images": total_images,
        "elapsed_time_sec": elapsed_time,
        "throughput_fps": throughput_fps,
        "latency_per_batch_ms": latency_per_batch_ms,
        "peak_vram_mb": peak_vram_mb
    }

def print_hardware_profile():
    print("=" * 75)
    print("           HARDWARE & COMPUTATIONAL RESOURCE PROFILER")
    print("=" * 75)
    
    # CPU & RAM
    cpu_ram = get_cpu_and_ram_info()
    print("\n[+] HOST CPU & SYSTEM RAM METRICS:")
    print(f"  * Operating System:          {cpu_ram.get('os')}")
    print(f"  * Logical CPU Cores:         {cpu_ram.get('cpu_cores_logical')} cores")
    if isinstance(cpu_ram.get('cpu_usage_pct'), (int, float)):
        print(f"  * Current CPU Utilization:   {cpu_ram.get('cpu_usage_pct')}%")
    if cpu_ram.get('ram_total_gb') != "N/A":
        print(f"  * System RAM (Total / Used): {cpu_ram.get('ram_used_gb'):.2f} GB / {cpu_ram.get('ram_total_gb'):.2f} GB ({cpu_ram.get('ram_usage_pct')}%)")
        print(f"  * System RAM Available:      {cpu_ram.get('ram_free_gb'):.2f} GB")

    # GPU
    gpu = get_gpu_info()
    print("\n[+] NVIDIA GPU & VRAM METRICS:")
    if not gpu.get("cuda_available"):
        print("  [!] CUDA is not available on this system.")
    else:
        print(f"  * GPU Device Name:           {gpu.get('gpu_name')}")
        print(f"  * CUDA Version / cuDNN:      CUDA {gpu.get('cuda_version')} / cuDNN {gpu.get('cudnn_version')}")
        print(f"  * Total Dedicated VRAM:      {gpu.get('vram_total_gb'):.2f} GB ({gpu.get('vram_total_mb'):,.1f} MB)")
        print(f"  * VRAM Currently Allocated:  {gpu.get('vram_allocated_mb'):.2f} MB")
        print(f"  * VRAM Reserved (Cached):    {gpu.get('vram_reserved_mb'):.2f} MB")
        print(f"  * VRAM Free Capacity:        {gpu.get('vram_free_mb'):,.1f} MB")
        if gpu.get('gpu_util_pct') != "N/A":
            print(f"  * GPU Core Utilization:      {gpu.get('gpu_util_pct')}%")
            print(f"  * GPU Temperature:           {gpu.get('gpu_temp_c')} °C")
            print(f"  * GPU Power Draw:            {gpu.get('gpu_power_w')} W")

    # Live Benchmark
    print("\n[+] RUNNING LIVE GPU THROUGHPUT BENCHMARK (3-Ch 256x256 ResNet)...")
    bench = run_live_gpu_benchmark(batch_size=32, num_batches=30)
    if bench:
        print(f"  * Processed:                 {bench['total_images']:,} images in {bench['elapsed_time_sec']:.2f} seconds")
        print(f"  * Pure GPU Throughput:       {bench['throughput_fps']:,.1f} images / second")
        print(f"  * Latency per Batch (B=32):  {bench['latency_per_batch_ms']:.1f} ms / batch")
        print(f"  * Peak VRAM During Pass:     {bench['peak_vram_mb']:.2f} MB")
    
    print("\n" + "=" * 75)
    print("                    PROFILING AUDIT COMPLETE")
    print("=" * 75)

if __name__ == "__main__":
    print_hardware_profile()
