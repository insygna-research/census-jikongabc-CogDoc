import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    # 测试进程需优先导入仓库源码。
    sys.path.insert(0, ROOT)
