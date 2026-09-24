"""
PyInstaller 启动入口 —— AI 知识库
负责：路径设置 → 启动 uvicorn → 打开浏览器
"""
import sys
import os
import shutil
import webbrowser
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python 3.10+ is required by LangChain 1.x; current version: {sys.version.split()[0]}"
    )

# ── 禁用遥测 ──
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "false")
os.environ.setdefault("CHROMA_ANONYMIZED_TELEMETRY", "false")
os.environ.setdefault("CHROMA_TELEMETRY_DISABLED", "1")

# ── 日志：所有 print 同时写入文件，方便诊断闪退 ──
LOG_FILE = None

def log(msg, end="\n"):
    try:
        print(msg, end=end)
    except UnicodeEncodeError:
        # GBK 终端不支持部分 Unicode（如 emoji），降级为 ASCII
        safe = msg.encode("ascii", errors="replace").decode("ascii")
        print(safe, end=end)
    if LOG_FILE:
        try:
            LOG_FILE.write(str(msg) + end)
            LOG_FILE.flush()
        except Exception:
            pass

# ── 路径策略 ──
# 打包模式 (onefile=True)：
#   sys._MEIPASS  = 临时解压目录（含 frontend/、.env）
#   EXE_DIR       = EXE 所在目录（持久数据存放处，如 chroma_db/、uploads/）
# 开发模式：
#   PROJECT_DIR   = 项目根目录

if getattr(sys, 'frozen', False):
    # ── 打包模式 ──
    MEIPASS = Path(sys._MEIPASS)           # 解压临时目录
    EXE_DIR = Path(sys.executable).parent.resolve()  # EXE 所在目录
    DATA_DIR = EXE_DIR                     # 持久数据与 EXE 同级

    # 设置日志文件
    try:
        LOG_FILE = open(EXE_DIR / "ai-kb.log", "a", encoding="utf-8")
    except Exception:
        pass

    log(f"[启动] AI 知识库 v2.1.0 (单文件模式)")
    log(f"[路径] EXE 目录: {EXE_DIR}")
    log(f"[路径] 临时资源: {MEIPASS}")

    # 切换到 EXE 目录（持久数据：uploads/、chroma_db/、categories.json、config.json）
    os.chdir(EXE_DIR)
    os.environ["APP_BASE_DIR"] = str(EXE_DIR)

    # 将 MEIPASS 加入 sys.path
    sys.path.insert(0, str(MEIPASS))
    sys.path.insert(0, str(EXE_DIR))

else:
    # ── 开发模式 ──
    PROJECT_DIR = Path(__file__).parent.resolve()
    os.chdir(PROJECT_DIR)
    os.environ["APP_BASE_DIR"] = str(PROJECT_DIR)
    sys.path.insert(0, str(PROJECT_DIR))
    log(f"[启动] AI 知识库 v2.1.0 (开发模式)")
    log(f"[路径] 项目目录: {PROJECT_DIR}")

# ── 导入 app ──
try:
    from backend.main import app
    log("[导入] app 加载成功")
except Exception as e:
    import traceback
    log(f"❌ 导入失败: {e}")
    traceback.print_exc()
    if LOG_FILE:
        traceback.print_exc(file=LOG_FILE)
    input("\n按 Enter 键退出...")
    sys.exit(1)

# ── 启动服务器 ──
import uvicorn
import socket
import subprocess
import time

HOST = os.getenv("APP_HOST", "0.0.0.0")
PORT = int(os.getenv("APP_PORT", "8000"))

# ── 检查端口 ──
def _find_and_kill_port_process(port):
    try:
        result = subprocess.run(
            f"netstat -ano | findstr :{port}",
            shell=True, capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 5:
                pid = parts[-1]
                try:
                    os.kill(int(pid), 9)
                    log(f"  ✓ 已停止旧进程 (PID: {pid})")
                except (OSError, ValueError):
                    pass
    except Exception:
        pass

def _is_port_available(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            return s.connect_ex(('127.0.0.1', port)) != 0
    except Exception:
        return True

if not _is_port_available(PORT):
    log(f"  ⚠️ 端口 {PORT} 已被占用，正在释放...")
    _find_and_kill_port_process(PORT)
    time.sleep(1)
    if not _is_port_available(PORT):
        log(f"  ⚠️ 端口 {PORT} 仍被占用，尝试 {PORT+1}...")
        PORT += 1

log(f"╔═══════════════════════════════════════════╗")
log(f"║      AI 知识库服务                         ║")
log(f"╠═══════════════════════════════════════════╣")
log(f"║  地址: http://localhost:{PORT}")
log(f"║  按 Ctrl+C 关闭                           ║")
log(f"╚═══════════════════════════════════════════╝")
log("")

# 打开浏览器
try:
    webbrowser.open(f"http://localhost:{PORT}")
except Exception as e:
    log(f"⚠️ 自动打开浏览器失败: {e}")
    log(f"   请手动访问 http://localhost:{PORT}")

# 启动 uvicorn
try:
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
    )
except Exception as e:
    import traceback
    log(f"\n❌ 服务异常退出: {e}")
    traceback.print_exc()
    if LOG_FILE:
        traceback.print_exc(file=LOG_FILE)
    input("\n按 Enter 键退出...")

# 关闭日志
if LOG_FILE:
    try:
        LOG_FILE.close()
    except Exception:
        pass
