"""Environment check for the CNN-Based Gesture Controlled Gaming Application.

Verifies the Python version, every required library, CUDA GPU acceleration through
PyTorch, and webcam availability. Prints a PASS/FAIL line per check.

CUDA is mandatory for this project: CNN training and inference must run on the GPU,
so a missing or broken CUDA setup is reported as a FAIL, never silently tolerated.

Usage:
    python src/environment_check.py
    python src/environment_check.py --skip-webcam
"""

import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

MIN_PYTHON = (3, 9)
MAX_PYTHON = (3, 12)

results = []


def record(name, passed, detail):
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name:<18} {detail}")


def check_python():
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    ok = MIN_PYTHON <= (v.major, v.minor) <= MAX_PYTHON
    detail = version if ok else f"{version} (expected {MIN_PYTHON[0]}.{MIN_PYTHON[1]}-{MAX_PYTHON[0]}.{MAX_PYTHON[1]})"
    record("Python", ok, detail)


def check_import(name, module_name, version_attr="__version__"):
    try:
        module = __import__(module_name)
        version = getattr(module, version_attr, "unknown")
        record(name, True, f"version {version}")
        return module
    except Exception as exc:
        record(name, False, f"import failed: {exc}")
        return None


def check_torch_and_cuda():
    """Import PyTorch, then verify CUDA is present, allocatable and actually computes."""
    torch = check_import("PyTorch", "torch")
    check_import("torchvision", "torchvision")
    if torch is None:
        record("CUDA available", False, "skipped, PyTorch unavailable")
        return

    print(f"       built against CUDA {torch.version.cuda}")

    if not torch.cuda.is_available():
        record("CUDA available", False, "torch.cuda.is_available() is False - GPU build or driver missing")
        return
    record("CUDA available", True, f"True (CUDA runtime {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()})")

    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    record("GPU", True, f"{name} | {vram_gb:.2f} GB VRAM | compute capability {props.major}.{props.minor}")

    # Allocation and a real computation on the device, not just a capability flag.
    try:
        device = torch.device("cuda")
        a = torch.randn(512, 512, device=device)
        b = torch.randn(512, 512, device=device)
        record("CUDA allocation", True, f"allocated {torch.cuda.memory_allocated() / 1024**2:.1f} MB on device 0")

        c = a @ b
        torch.cuda.synchronize()
        assert c.device.type == "cuda", "result tensor did not stay on the GPU"
        assert c.shape == (512, 512)
        record("CUDA computation", True, f"512x512 matmul on {c.device}, result mean {c.mean().item():.4f}")

        del a, b, c
        torch.cuda.empty_cache()
    except Exception as exc:
        record("CUDA computation", False, f"GPU computation failed: {exc}")


def check_pygame():
    pygame = check_import("Pygame", "pygame")
    if pygame is None:
        return
    try:
        pygame.init()
        pygame.quit()
        record("Pygame init", True, "initialized and shut down cleanly")
    except Exception as exc:
        record("Pygame init", False, f"initialization failed: {exc}")


def check_webcam():
    try:
        import cv2
    except Exception:
        record("Webcam", False, "skipped, OpenCV unavailable")
        return

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
    try:
        if not capture.isOpened():
            record("Webcam", False, "could not open the default webcam (index 0)")
            return
        ok, frame = capture.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            record("Webcam", True, f"captured a {width}x{height} frame from device 0")
        else:
            record("Webcam", False, "webcam opened but no frame was returned")
    finally:
        capture.release()


def main():
    print("Environment check - CNN-Based Gesture Controlled Gaming Application")
    print("-" * 74)

    check_python()
    check_torch_and_cuda()
    check_import("OpenCV", "cv2")
    check_pygame()
    check_import("NumPy", "numpy")
    check_import("Matplotlib", "matplotlib")
    check_import("scikit-learn", "sklearn")

    if "--skip-webcam" in sys.argv:
        print("[SKIP] Webcam             skipped via --skip-webcam")
    else:
        check_webcam()

    print("-" * 74)
    failed = [name for name, passed, _ in results if not passed]
    if failed:
        print(f"RESULT: FAIL ({len(failed)} of {len(results)} checks failed: {', '.join(failed)})")
        return 1
    print(f"RESULT: PASS (all {len(results)} checks passed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
