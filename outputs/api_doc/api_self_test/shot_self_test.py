"""接口自测与截图（API 文档验收素材）：真实起 Flask 服务 → 逐用例真实 HTTP 调用 → 截图。

自测工具 = 浏览器 HTTP 客户端面板（outputs/api_doc/api_self_test/live_panel.html，与 Postman 同类）：
面板对每个用例向 http://127.0.0.1:PORT 真实发起一次请求并渲染真实响应；
本脚本用无头 Chrome 逐用例截图，并另用 urllib 把真实响应 JSON 存盘供文档引用。

运行：venv\\Scripts\\python.exe outputs\\api_doc\\api_self_test\\shot_self_test.py
依赖：Chrome / Edge（找不到会报错）；模型与数据产物已就绪（models/、data/processed/）。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[3]           # d:/0000/recommender_project
SELF = Path(__file__).resolve().parent
SHOTS = SELF / "shots";  SHOTS.mkdir(parents=True, exist_ok=True)
RESPS = SELF / "responses"; RESPS.mkdir(parents=True, exist_ok=True)

PORT = 5099
BASE = f"http://127.0.0.1:{PORT}"
PANEL = (SELF / "live_panel.html").as_uri()

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
    raise SystemExit("未找到 Chrome/Edge")


# 用例：(名字, 方法, 相对路径, 期望从渲染页抓到的片段, 截图高px)。示例一律 topn=3 便于整段响应上屏。
CASES = [
    ("T1_health",          "GET", "/health",                                        '"status": "ok"',                     760),
    ("T2_similar_i2v",     "GET", "/similar?item_id=85123A&topn=3&method=item2vec&format=json",  '"item_id": "85123A"',     1380),
    ("T3_similar_itemcf",  "GET", "/similar?item_id=85123A&topn=3&method=itemcf&format=json",    '"method": "itemcf"',       1380),
    ("T4_recommend_lr",    "GET", "/recommend?user_id=12597&topn=3&ranker=lr&format=json",       '"ranker": "lr"',           1660),
    ("T5_recommend_lgb",   "GET", "/recommend?user_id=12597&topn=3&ranker=lgb&format=json",      '"ranker": "lgb"',          1660),
    ("T6_baseline",        "GET", "/baseline?user_id=13899&topn=3&format=json",                  '"columns"',                1960),
    ("E1_similar_no_item", "GET", "/similar?format=json",                              "缺少 item_id 参数",               920),
    ("E2_similar_cold",    "GET", "/similar?item_id=ZZZZ9&method=item2vec&format=json", "不在嵌入词表",                 920),
    ("E3_recommend_nouser","GET", "/recommend?user_id=99999999&ranker=lr&format=json",  "用户 99999999 不存在",         920),
    ("E4_recommend_ranker","GET", "/recommend?user_id=12597&ranker=svm&format=json",    "ranker 只能是 lr 或 lgb",     920),
]


def wait_ready(timeout: float = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as r:
                if json.loads(r.read().decode("utf-8")).get("ready"):
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError("Flask 服务未能就绪")


def chrome(extra: list[str], url: str, win: tuple[int, int] = (1280, 900), timeout: int = 120) -> subprocess.CompletedProcess:
    user_data = tempfile.mkdtemp(prefix="chrome_apidoc_")
    cmd = [str(find_chrome()), "--headless=new", "--disable-gpu", "--hide-scrollbars",
           "--no-sandbox", "--disable-dev-shm-usage", "--allow-file-access-from-files",
           "--disable-web-security", f"--user-data-dir={user_data}",
           f"--window-size={win[0]},{win[1]}", *extra, url]
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def panel_url(rel: str, name: str) -> str:
    q = urllib.parse.urlencode({"base": BASE, "rel": rel, "name": name, "method": "GET"})
    return f"{PANEL}?{q}"


def http_get(rel: str) -> tuple[int, str, dict]:
    url = BASE + rel
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8")
        return r.status, body, dict(r.headers)


def main() -> None:
    env = dict(os.environ)
    env["PORT"] = str(PORT)
    logf = (SELF / "server.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "web" / "app.py")], env=env,
        stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT), text=True, encoding="utf-8",
    )
    summary = []
    try:
        print("等待 Flask 就绪 …")
        wait_ready()
        print(f"就绪：{BASE}，共 {len(CASES)} 个用例\n")
        for name, method, rel, snippet, height in CASES:
            # ① 真实 HTTP 调用，响应落盘
            try:
                status, body, headers = http_get(rel)
            except Exception as e:
                status, body, headers = 0, f"请求异常: {e}", {}
            pretty = body
            try:
                pretty = json.dumps(json.loads(body), ensure_ascii=False, indent=2)
            except Exception:
                pass
            (RESPS / f"{name}.json").write_text(pretty, encoding="utf-8")
            ctype = headers.get("Content-Type", "")

            # ② 无头 Chrome 渲染面板页（面板内对该接口真实 fetch），先 dump-dom 校验再截图
            url = panel_url(rel, name)
            dom = chrome(["--virtual-time-budget=10000", "--dump-dom"], url)
            dom_text = dom.stdout.decode("utf-8", errors="replace")
            found = snippet in dom_text
            shot = chrome(["--virtual-time-budget=10000",
                           f"--screenshot={SHOTS / (name + '.png')}"], url, win=(980, height))
            png = SHOTS / f"{name}.png"
            kb = png.stat().st_size / 1024 if png.exists() else 0
            mark = "OK " if (found and png.exists() and kb > 4) else "FAIL"
            print(f"[{mark}] {name:<22} HTTP {status:<3} 渲染片段命中={found}  截图 {kb:6.1f} KB")
            summary.append({"case": name, "method": method, "url": BASE + rel,
                            "status": status, "content_type": ctype, "rendered": found, "screenshot_kb": round(kb, 1)})
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=6)
        except Exception:
            proc.kill()
        logf.close()

    (SELF / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n汇总 -> {SELF / 'summary.json'}；截图 -> {SHOTS}")


if __name__ == "__main__":
    main()
