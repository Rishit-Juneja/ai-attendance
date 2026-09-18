"""
Package init exists to do one thing: make cuDNN findable before any ONNX session
is created.

The nvidia pip wheels ship libcudnn.so.9 but no unversioned libcudnn.so, which is
the name onnxruntime dlopen()s. Without this, session creation still reports
CUDAExecutionProvider and then dies at the first Conv node with "cuDNN is
unavailable or disabled" — or silently falls back to CPU, which is worse because
nothing tells you.

It lives here rather than in pipeline.py or persons.py because both create
sessions, and whichever happened to import first would otherwise own it.
"""

try:
    import onnxruntime as _ort

    if hasattr(_ort, "preload_dlls"):
        _ort.preload_dlls()
except Exception:  # noqa: BLE001 - CPU-only installs have nothing to preload
    pass
