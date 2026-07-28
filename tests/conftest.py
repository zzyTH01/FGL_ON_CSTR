import os

# 必须在 fgl_common 被 import 之前设置,使 training.device 解析为 CPU。
os.environ.setdefault("FGL_DEVICE", "cpu")
