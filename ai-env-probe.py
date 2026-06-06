import sys
import os
import platform
import subprocess
import json
import argparse

def get_pip_version():
    try:
        res = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            return res.stdout.strip()
    except Exception:
        pass
    return "Unknown"

def get_ram_info():
    if platform.system() == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong)
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullTotalPhys / (1024**3), 2)
        except Exception:
            pass
    # Fallback to psutil if installed/available
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass
    return None

def get_nvidia_smi_info():
    nvidia_smi_available = False
    gpus = []
    driver_version = None
    cuda_driver_version = None
    nvidia_smi_error = None

    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            nvidia_smi_available = True
            
            # Parse driver and CUDA versions from standard nvidia-smi screen
            for line in res.stdout.split("\n"):
                if "Driver Version" in line:
                    parts = line.split("Driver Version:")
                    if len(parts) > 1:
                        driver_version = parts[1].strip().split()[0]
                    if "CUDA Version" in line:
                        cuda_part = line.split("CUDA Version:")
                        if len(cuda_part) > 1:
                            cuda_driver_version = cuda_part[1].strip().split()[0]
                            cuda_driver_version = cuda_driver_version.replace("|", "").strip()

            # Parse GPUs
            query_res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if query_res.returncode == 0:
                for line in query_res.stdout.strip().split("\n"):
                    if not line.strip():
                        continue
                    parts = line.split(",")
                    name = parts[0].strip()
                    vram_str = parts[1].strip() if len(parts) > 1 else "Unknown"
                    try:
                        vram_val = int(vram_str)
                    except ValueError:
                        vram_val = vram_str
                    gpus.append({
                        "name": name,
                        "vram_total_mb": vram_val
                    })
        else:
            nvidia_smi_error = res.stderr.strip()
    except Exception as e:
        nvidia_smi_error = str(e)

    return {
        "available": nvidia_smi_available,
        "gpus": gpus,
        "driver_version": driver_version,
        "cuda_version": cuda_driver_version,
        "error": nvidia_smi_error
    }

def get_nvcc_info():
    nvcc_available = False
    nvcc_version = None
    try:
        res = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            nvcc_available = True
            for line in res.stdout.split("\n"):
                if "release" in line:
                    nvcc_version = line.strip()
                    break
            if not nvcc_version:
                nvcc_version = res.stdout.strip()
    except Exception:
        pass
    return {
        "available": nvcc_available,
        "version": nvcc_version
    }

def run_torch_cuda_test():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "skipped", "message": "CUDA not available"}
        
        # Tiny tensor test
        t1 = torch.tensor([1.0, 2.0], device="cuda")
        t2 = t1 * 2.0
        val = t2.cpu().tolist()
        if val == [2.0, 4.0]:
            return {"status": "success", "result": val}
        else:
            return {"status": "failed", "result": val, "error": f"Unexpected computation result: {val}"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="AI-first Environment Probe for YOLO-Lab")
    parser.add_argument("--json", action="store_true", help="Output exact JSON data for program parsing")
    args = parser.parse_args()

    warnings = []
    errors = []

    # 1. Python info
    is_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    python_info = {
        "executable": sys.executable,
        "version": platform.python_version(),
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "is_venv": is_venv,
        "pip_version": get_pip_version()
    }

    # 2. System info
    system_info = {
        "os": platform.system(),
        "release": platform.release(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gb": get_ram_info()
    }

    # 3. NVIDIA SMI info
    smi_info = get_nvidia_smi_info()
    nvidia_info = {
        "available": smi_info["available"],
        "gpu_count": len(smi_info["gpus"]),
        "gpus": smi_info["gpus"],
        "error": smi_info["error"]
    }
    if smi_info["error"]:
        warnings.append(f"nvidia-smi query warning: {smi_info['error']}")

    # 4. CUDA Driver / Compiler info
    nvcc = get_nvcc_info()
    cuda_path = os.environ.get("CUDA_PATH")
    cuda_path_env_vars = {k: v for k, v in os.environ.items() if k.startswith("CUDA_PATH")}
    
    cuda_driver_info = {
        "driver_version": smi_info["driver_version"],
        "cuda_version_supported": smi_info["cuda_version"],
        "nvcc_available": nvcc["available"],
        "nvcc_version": nvcc["version"],
        "cuda_path": cuda_path,
        "cuda_path_env_vars": cuda_path_env_vars
    }

    # 5. Packages import and version detection
    torch_info = {
        "available": False,
        "version": None,
        "cuda_available": False,
        "compiled_cuda_version": None,
        "device_count": 0,
        "devices": [],
        "tensor_test": None
    }
    try:
        import torch
        torch_info["available"] = True
        torch_info["version"] = torch.__version__
        try:
            torch_info["cuda_available"] = torch.cuda.is_available()
            if torch_info["cuda_available"]:
                torch_info["compiled_cuda_version"] = torch.version.cuda
                torch_info["device_count"] = torch.cuda.device_count()
                torch_info["devices"] = [torch.cuda.get_device_name(i) for i in range(torch_info["device_count"])]
                torch_info["tensor_test"] = run_torch_cuda_test()
        except Exception as ce:
            warnings.append(f"torch.cuda check failed: {ce}")
    except Exception as e:
        warnings.append(f"torch package is not available or failed to import: {e}")

    onnx_info = {"available": False, "version": None}
    try:
        import onnx
        onnx_info["available"] = True
        onnx_info["version"] = onnx.__version__
    except Exception as e:
        warnings.append(f"onnx package is not available or failed to import: {e}")

    onnxruntime_info = {
        "available": False,
        "version": None,
        "providers": [],
        "cuda_provider_available": False
    }
    try:
        import onnxruntime
        onnxruntime_info["available"] = True
        onnxruntime_info["version"] = onnxruntime.__version__
        try:
            onnxruntime_info["providers"] = onnxruntime.get_available_providers()
            onnxruntime_info["cuda_provider_available"] = "CUDAExecutionProvider" in onnxruntime_info["providers"]
        except Exception as oe:
            warnings.append(f"onnxruntime providers check failed: {oe}")
    except Exception as e:
        warnings.append(f"onnxruntime package is not available or failed to import: {e}")

    ultralytics_info = {"available": False, "version": None}
    try:
        import ultralytics
        ultralytics_info["available"] = True
        ultralytics_info["version"] = ultralytics.__version__
    except Exception as e:
        warnings.append(f"ultralytics package is not available or failed to import: {e}")

    opencv_info = {"available": False, "version": None}
    try:
        import cv2
        opencv_info["available"] = True
        opencv_info["version"] = cv2.__version__
    except Exception as e:
        warnings.append(f"opencv (cv2) package is not available or failed to import: {e}")

    # Final payload structure
    payload = {
        "python": python_info,
        "system": system_info,
        "nvidia": nvidia_info,
        "cuda_driver": cuda_driver_info,
        "torch": torch_info,
        "onnx": onnx_info,
        "onnxruntime": onnxruntime_info,
        "ultralytics": ultralytics_info,
        "opencv": opencv_info,
        "warnings": warnings,
        "errors": errors
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        # Human-friendly short summary
        print("=" * 70)
        print(" YOLO-Lab Environment Probe Summary")
        print("=" * 70)
        print(f"Python Executable: {python_info['executable']}")
        print(f"Python Version:    {python_info['version']} (venv: {python_info['is_venv']})")
        print(f"OS Platform:       {system_info['os']} {system_info['release']} ({system_info['architecture']})")
        print(f"CPU Count / RAM:   {system_info['cpu_count']} Cores / {system_info['ram_gb']} GB")
        
        gpu_status = "Not Available"
        if nvidia_info["available"]:
            gpu_status = ", ".join([gpu["name"] for gpu in nvidia_info["gpus"]])
        print(f"NVIDIA GPU(s):     {gpu_status}")
        
        nvcc_status = nvcc["version"] if nvcc["available"] else "Not Available"
        print(f"NVCC (CUDA Compiler): {nvcc_status}")
        print(f"CUDA_PATH Env Var:    {cuda_driver_info['cuda_path']}")
        
        print("\nPackages Status:")
        print(f"  torch:           {torch_info['version'] if torch_info['available'] else 'Not Installed'} (CUDA: {torch_info['cuda_available']})")
        print(f"  onnx:            {onnx_info['version'] if onnx_info['available'] else 'Not Installed'}")
        print(f"  onnxruntime:     {onnxruntime_info['version'] if onnxruntime_info['available'] else 'Not Installed'} (CUDA provider: {onnxruntime_info['cuda_provider_available']})")
        print(f"  ultralytics:     {ultralytics_info['version'] if ultralytics_info['available'] else 'Not Installed'}")
        print(f"  opencv (cv2):    {opencv_info['version'] if opencv_info['available'] else 'Not Installed'}")
        
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")
        if errors:
            print("\nErrors:")
            for e in errors:
                print(f"  - {e}")
        print("=" * 70)

if __name__ == "__main__":
    main()
