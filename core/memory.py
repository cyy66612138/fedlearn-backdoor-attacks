"""
Memory monitoring and optimization utilities
"""

import torch
import gc
import psutil
import subprocess
import os
from typing import Tuple


class MemoryMonitor:
    """Advanced memory monitoring and optimization"""
    
    _baseline = None
    _gpu_id = None
    _peak_cpu = 0.0
    _peak_gpu = 0.0
    
    @classmethod
    def _gpu(cls):
        return cls._gpu_id or (torch.cuda.current_device() if torch.cuda.is_available() else 0)
    
    @classmethod
    def set_gpu_id(cls, gpu_id): 
        cls._gpu_id = gpu_id
    
    @classmethod
    def reset_peaks(cls):
        cls._peak_cpu = cls._peak_gpu = 0.0
    
    @classmethod
    def update_peaks(cls):
        cpu_used, *_ = cls.get_cpu_memory_relative()
        gpu_used, *_ = cls.get_nvidia_smi_memory()
        cls._peak_cpu = max(cls._peak_cpu, cpu_used)
        cls._peak_gpu = max(cls._peak_gpu, gpu_used)
    
    @classmethod
    def get_round_peaks(cls):
        return cls._peak_cpu, cls._peak_gpu
    
    @classmethod
    def get_nvidia_smi_memory(cls):
        try:
            cmd = [
                'nvidia-smi', f'--id={cls._gpu()}',
                '--query-gpu=memory.used,memory.total,memory.free',
                '--format=csv,noheader,nounits'
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
            used, total, free = map(int, out.split(', '))
            return used/1024, total/1024, free/1024
        except Exception:
            return 0.0, 0.0, 0.0
    
    @classmethod
    def get_gpu_memory(cls):
        if not torch.cuda.is_available(): 
            return 0.0, 0.0, 0.0
        torch.cuda.set_device(cls._gpu())
        return (
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9,
            torch.cuda.max_memory_allocated() / 1e9
        )
    
    @staticmethod
    def _cpu_mem():
        m = psutil.virtual_memory()
        return m.used / 1e9, m.available / 1e9, m.percent
    
    @classmethod
    def init_cpu_baseline(cls):
        cls._baseline = cls._cpu_mem()
        u, a, p = cls._baseline
        print(f"🔧 CPU Baseline: used={u:.2f}GB, available={a:.2f}GB, {p:.1f}%")
    
    @classmethod
    def get_cpu_memory_relative(cls):
        now = cls._cpu_mem()
        if cls._baseline is None:
            return (0.0, 0.0, 0.0, *now)
        diffs = [now[i] - cls._baseline[i] for i in range(3)]
        return (*diffs, *now)
    
    @classmethod
    def monitor_memory(cls, stage="Memory", verbose=True):
        gpu_id = cls._gpu()
        phys_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', gpu_id))
        
        smi_used, smi_total, smi_free = cls.get_nvidia_smi_memory()
        alloc, reserved, peak = cls.get_gpu_memory()
        d_used, d_avail, d_pct, used, avail, pct = cls.get_cpu_memory_relative()
        
        cls.update_peaks()
        
        if verbose:
            print(f"📊 {stage}:")
            print(f"   🐍 GPU [nvidia-smi:{phys_id}] {smi_used:.2f}/{smi_total:.2f}GB used, {smi_free:.2f}GB free")
            print(f"   🐍 GPU [torch] alloc={alloc:.2f}, reserved={reserved:.2f}, peak={peak:.2f} GB")
            print(f"   💻 CPU: {used:.2f}GB used ({pct:.1f}%) [Δ{d_used:+.2f}GB, Δ{d_pct:+.1f}%]")
        
        return {
            'gpu_id': gpu_id,
            'gpu_physical_id': phys_id,
            'gpu_used_smi': smi_used,
            'gpu_total_smi': smi_total,
            'gpu_free_smi': smi_free,
            'gpu_allocated': alloc,
            'gpu_reserved': reserved,
            'gpu_peak': peak,
            'cpu_used': used,
            'cpu_available': avail,
            'cpu_percent': pct,
            'cpu_used_diff': d_used,
            'cpu_available_diff': d_avail,
            'cpu_percent_diff': d_pct
        }
    
    @staticmethod
    def cleanup_memory(aggressive=False):
        gc.collect()
        if torch.cuda.is_available():
            for _ in range(3 if aggressive else 1):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
