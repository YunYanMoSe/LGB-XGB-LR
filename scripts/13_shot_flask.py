"""阶段六 · Flask 服务截图（无头 Chrome 抓取 5 张演示页）。

自动：以 PORT=5052 起独立 Flask 进程 → 等 /health 就绪 →
对演示主页 / 相似 / 精排(LR、LGB) / 基线 各截一张 → 关进程。
截图落 outputs/chain/flask_*.png（供 Word 素材引用）。

运行：venv\\Scripts\\python.exe scripts\\13_shot_flask.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PORT = 5052
BASE = f"http://127.0.0.1:{PORT}"
CHAIN = PROJECT_ROOT / "outputs" / "chain"
CHAIN.mkdir(parents=True, exist_ok=True)

# 常见 Chrome 路径（找不到就报错）
CHROME_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
    Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
    Path("C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"),
    Path("C:/Program Files/Microsoft/Edge/Application/msedge.exe"),
]


def find_chrome() -> Path:
    for p in CHROME_CANDIDATES:
        if p.exists():
            return p
    raise SystemExit("未找到 Chrome/Edge，请手动截图 outputs/chain/flask_*.png 对应页面")


def wait_ready(timeout: float = 60) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.4)
    raise TimeoutError("Flask 服务未能就绪")


def demo_ids() -> tuple[str, str]:
    p = PROJECT_ROOT / "models" / "demo_ids.json"
    if p.exists():
        j = json.loads(p.read_text(encoding="utf-8"))
        if j.get("users"):
            return str(j["users"][0]), str(j["users"][-1])
    return "12597", "13899"


def shot(chrome: Path, url: str, name: str, width: int = 1280, height: int = 1100) -> Path:
    out = CHAIN / name
    user_data = Path(tempfile.mkdtemp(prefix="chrome_shot_"))
    cmd = [
        str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--no-sandbox", "--disable-dev-shm-usage",
        f"--user-data-dir={user_data}",
        f"--window-size={width},{height}",
        f"--screenshot={out}", "--virtual-time-budget=5000", url,
    ]
    subprocess.run(cmd, check=False, capture_output=True, timeout=120)
    if not out.exists() or out.stat().st_size < 5000:
        raise RuntimeError(f"截图失败或为空：{name} ({out.stat().st_size if out.exists() else 0} bytes)")
    print(f"  ✓ {name}  {out.stat().st_size/1024:.0f} KB  ←  {url}")
    return out


def main() -> None:
    chrome = find_chrome()
    user, user2 = demo_ids()
    env = dict(os.environ)
    env["PORT"] = str(PORT)
    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "web" / "app.py")], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8",
    )
    try:
        print("等待 Flask 就绪…")
        wait_ready()
        print("开始截图：")
        shot(chrome, f"{BASE}/", "flask_home.png")
        shot(chrome, f"{BASE}/similar?item_id=85123A&topn=5", "flask_similar.png")
        shot(chrome, f"{BASE}/recommend?user_id={user}&topn=10&ranker=lr", "flask_recommend_lr.png")
        shot(chrome, f"{BASE}/recommend?user_id={user2}&topn=10&ranker=lgb", "flask_recommend_lgb.png")
        shot(chrome, f"{BASE}/baseline?user_id={user}&topn=10", "flask_baseline.png")
        print(f"\n全部截图见 {CHAIN}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    main()
