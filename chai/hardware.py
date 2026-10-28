"""The machine a local model ran on, recorded with every local call.

A hosted model's cost is its bill; a local model's cost is the machine and the time. So every
in-process backend (``TransformersComponent``, ``MLXVLMComponent``, System One) and every served
backend whose server is on this machine (Ollama, LM Studio, a vLLM or sglang at localhost) puts a
small ``local_compute`` entry in the metadata of each result it makes -- the call's timing, its
device, and a key for the machine::

    {"hardware": "hw_5c1e0b9a42", "device": "mps", "seconds": 41.2, "hours": 0.011444}

The machine itself is described once, not on every result: the workflow keeps ``{key: profile}``
for the machines its run used (``Workflow.hardware``) and records it on the run, in the
WORKFLOW_RUN row's ``metadata.hardware`` that every storage writes (``run.records.json`` beside a
JSON-Lines file, a Parquet file's ``chai_tea_records``, the ``workflow_runs`` table)::

    {"hw_5c1e0b9a42": {"os": "Darwin", "os_version": "15.6", "machine": "arm64",
                       "cpu": "Apple M4 Max", "cpu_cores_physical": 16, "cpu_cores_logical": 16,
                       "ram_gb": 128.0,
                       "gpus": [{"vendor": "Apple", "name": "Apple M4 Max", "cores": 40, "memory_gb": 128.0}]}}

The key is a hash of the profile, so the same machine has the same key in every run and rows from
many runs join to one description. ``device`` is where the model's weights were (``cuda:0``,
``mps``, ``cpu``), or ``None`` for a local server, which decides that itself. The profile is read
once per process (``hardware_info``) and the same on Windows, macOS and Linux: the CPU's model name
from the registry, ``sysctl`` or ``/proc/cpuinfo``; NVIDIA GPUs through NVML (``nvidia-ml-py``) or,
failing that, torch; Apple Silicon's GPU from ``system_profiler``; other GPUs on Windows from CIM. A
probe that cannot run leaves its field ``None`` or its list empty rather than failing the call.
``Workflow.usage`` totals the hours (``local_seconds``, ``by_hardware``).
"""

import functools
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
from urllib.parse import urlparse

logger = logging.getLogger("chai")

LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _gb(n):
    return round(n / 1024**3, 1) if n else None


def cpu_name():
    """The CPU's model name, e.g. 'Apple M4 Max', 'AMD Ryzen 9 7950X 16-Core Processor'."""
    system = platform.system()
    name = ""
    if system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    elif system == "Linux":
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.lower().startswith(("model name", "hardware", "cpu model")):
                        name = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
    elif system == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except Exception:
            pass
    return name or platform.processor() or None


def ram_bytes():
    try:
        import psutil

        return psutil.virtual_memory().total
    except Exception:
        pass
    if hasattr(os, "sysconf"):
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError):
            pass
    return None


def cpu_cores():
    """``(physical, logical)`` core counts."""
    try:
        import psutil

        return psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)
    except Exception:
        return None, os.cpu_count()


def nvidia_gpus():
    """NVIDIA GPUs through NVML, with the driver version."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception:
        return []
    gpus = []
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            gpus.append(
                {
                    "vendor": "NVIDIA",
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "memory_gb": _gb(pynvml.nvmlDeviceGetMemoryInfo(handle).total),
                    "driver": driver,
                }
            )
    except Exception as e:
        logger.debug(f"NVML could not list GPUs: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    return gpus


def torch_gpus():
    """CUDA (or ROCm) devices as torch sees them, for machines without NVML."""
    torch = sys.modules.get("torch")
    if torch is None:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        vendor = "AMD" if getattr(torch.version, "hip", None) else "NVIDIA"
        return [
            {
                "vendor": vendor,
                "name": torch.cuda.get_device_name(i),
                "memory_gb": _gb(torch.cuda.get_device_properties(i).total_memory),
            }
            for i in range(torch.cuda.device_count())
        ]
    except Exception:
        return []


def apple_gpus(ram):
    """Apple Silicon's integrated GPU (it shares the machine's memory)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return []
    gpus = []
    try:
        data = json.loads(_run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=20) or "{}")
        for gpu in data.get("SPDisplaysDataType", []):
            cores = gpu.get("sppci_cores")
            gpus.append(
                {
                    "vendor": "Apple",
                    "name": gpu.get("sppci_model") or gpu.get("_name"),
                    "cores": int(cores) if str(cores or "").isdigit() else None,
                    "memory_gb": _gb(ram),
                }
            )
    except Exception:
        pass
    return gpus or [{"vendor": "Apple", "name": cpu_name(), "cores": None, "memory_gb": _gb(ram)}]


def windows_gpus():
    """Display adapters on Windows (any vendor), through CIM."""
    if platform.system() != "Windows":
        return []
    out = _run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json"],
        timeout=20,
    )
    try:
        data = json.loads(out) if out else []
    except ValueError:
        return []
    data = [data] if isinstance(data, dict) else data
    gpus = []
    for gpu in data:
        name = gpu.get("Name") or ""
        vendor = next((v for v in ("NVIDIA", "AMD", "Intel") if v.lower() in name.lower()), None)
        gpus.append({"vendor": vendor, "name": name, "memory_gb": _gb(gpu.get("AdapterRAM"))})
    return gpus


@functools.lru_cache(maxsize=1)
def hardware_info():
    """This machine: OS, CPU, cores, RAM and GPUs (read once per process)."""
    ram = ram_bytes()
    physical, logical = cpu_cores()
    system = platform.system()
    if system == "Darwin":
        os_version = platform.mac_ver()[0] or platform.release()
    elif system == "Windows":
        os_version = platform.version()
    else:
        os_version = platform.release()
    gpus = nvidia_gpus() or torch_gpus()
    if not gpus:
        gpus = apple_gpus(ram) or windows_gpus()
    return {
        "os": system,
        "os_version": os_version,
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "cpu_cores_physical": physical,
        "cpu_cores_logical": logical,
        "ram_gb": _gb(ram),
        "gpus": gpus,
    }


def hardware_label(hardware):
    """A short name for a machine, e.g. 'Apple M4 Max · 128.0 GB RAM · Darwin'."""
    hardware = hardware or {}
    gpus = [g.get("name") for g in hardware.get("gpus") or [] if g.get("name") and g.get("name") != hardware.get("cpu")]
    parts = [hardware.get("cpu") or "unknown CPU"] + gpus
    if hardware.get("ram_gb"):
        parts.append(f"{hardware['ram_gb']} GB RAM")
    parts.append(hardware.get("os") or "?")
    return " · ".join(parts)


def profile_key(profile):
    """A short, stable key for a hardware *profile*: the same machine gets the same key in every run."""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":"), default=str)
    return "hw_" + hashlib.sha256(canonical.encode()).hexdigest()[:10]


@functools.lru_cache(maxsize=1)
def hardware_key():
    """This machine's key (see ``profile_key``)."""
    return profile_key(hardware_info())


def hardware_profiles(keys=None):
    """``{key: profile}`` for the machines named by *keys* that this process knows -- this one."""
    profiles = {hardware_key(): hardware_info()}
    return profiles if keys is None else {k: v for k, v in profiles.items() if k in set(keys)}


def local_compute(seconds, device=None):
    """The ``local_compute`` metadata entry for a call that took *seconds* on this machine: the
    machine's key (its profile is kept once, on the run), the device and the time."""
    seconds = float(seconds or 0.0)
    return {
        "hardware": hardware_key(),
        "device": str(device) if device is not None else None,
        "seconds": round(seconds, 3),
        "hours": round(seconds / 3600, 6),
    }


def is_local_url(url):
    """True when *url* is a server on this machine."""
    if not url:
        return False
    host = urlparse(url if "//" in url else f"//{url}").hostname or ""
    return host.lower() in LOCAL_HOSTS


def torch_device(preferred=None):
    """The device to load a torch model on: *preferred* when given, else CUDA, then MPS, then CPU."""
    if preferred:
        return preferred
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
