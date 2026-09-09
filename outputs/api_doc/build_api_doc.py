"""把「接口文档素材 md」合成并渲染成 Word（API 接口文档）。

JSON 示例直接取自自测时真实响应（responses/*.json），逐行转等宽字块嵌入，
保证文档中的“正确/错误返回示例” = 服务实际返回原文。

运行：venv\\Scripts\\python.exe outputs\\api_doc\\build_api_doc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.md_to_docx import render  # noqa: E402

HERE = Path(__file__).resolve().parent
RESP = HERE / "api_self_test" / "responses"


def cb(name: str) -> str:
    """读取真实响应文件，逐行加反引号 → 渲染成 Consolas 等宽段落（保行结构）。"""
    pretty = (RESP / f"{name}.json").read_text(encoding="utf-8").rstrip()
    return "\n".join(f"`{ln}`" for ln in pretty.splitlines() if ln.strip())


def img(file: str, alt: str, note: str = "") -> str:
    s = f"![{alt}](api_self_test/shots/{file})\n"
    if note:
        s += f"> {note}\n"
    return s


TEMPLATE = r"""# 推荐系统召回→精排在线服务 · API 接口文档

> 服务类型：推荐/检索查询 API（只读）。本文档面向对接方（前端 / 测试 / 课程验收）。
> 作者：＿＿＿＿（学号／姓名）　·　日期：2026-09-05　·　接口版本：v1.0
> 验收要点：① 通信规范 ② 安全与加密 ③ 接口定义 ④ Postman 同类工具自测截图（第五节），可对照复现命令核验。

---

## 一、接口总览与通用约定

本服务把“多路召回（热门 10 / 相似扩展 20 / 用户向量 20 融合 50）→ LR / LightGBM 精排”链路封装为 HTTP 只读接口，另提供商品相似与基线对照。在线上下文基于全量清洗交易（4,338 位用户）与全量商品向量（词表 3,177），属静态历史知识，无任何用户评估标签。

| 端点 | Method | 说明 | JSON 入口 |
|---|---|---|---|
| `/health` | GET | 健康检查 / 模型就绪探针 | 恒为 JSON |
| `/` | GET | 演示主页（HTML 页面，非数据接口） | — |
| `/similar` | GET | 商品相似 Top-N（Item2Vec / ItemCF 两法） | 加 `&format=json` |
| `/recommend` | GET | 召回→精排 为用户推荐 Top-N | 加 `&format=json` |
| `/baseline` | GET | 基线对照（全局热门 / ItemCF / 融合召回） | 加 `&format=json` |

通用约定：

- 传输：HTTP/1.1；请求一律 `GET`，参数放在 URL 查询串（Query String）中。
- 数据接口返回 `application/json`（`/similar` `/recommend` `/baseline` 需带 `&format=json`；不带则返回给人看的 HTML 演示页）。`/health` 恒返回 JSON。
- 编码：全局 UTF-8（无 BOM）；JSON 键为英文，值为中文/英文混合展示文案。
- 字段风格：商品信息由 英文目录 + 中文译名/类目 合并而成（详见各接口返回字段表）。

---

## 二、通信规范

### 传输协议

| 场景 | 协议 | 说明 |
|---|---|---|
| 开发 / 联调 / 自测 | HTTP/1.1（本机回环） | 服务默认 `http://127.0.0.1:5001`，可用环境变量 `PORT` 覆盖；只绑定回环地址 |
| 生产 / 验收部署 | HTTPS（TLS 1.2 及以上） | 由反向代理（Nginx 等）终结 TLS 后再转发给 Flask；对外禁止明文 HTTP |
| 版本 | HTTP/1.1 | Flask（Werkzeug）开发服务器 / 生产 WSGI（gunicorn）均支持 |

启动与绑定：

- 本地：`venv\Scripts\python.exe web\app.py` → `http://127.0.0.1:5001`（加载模型约需数十秒，`/health` 的 `ready` 由 `false` 翻转为 `true` 表示就绪）。
- 换端口：设置环境变量 `PORT=8080` 后启动。
- 生产：`gunicorn -w 2 -b 127.0.0.1:8000 web.app:app`，前端再经 Nginx 的 `server { listen 443 ssl; ... }` 提供 HTTPS。

### 请求 / 响应格式

请求：全部为 `GET`，参数为键值对，值需 URL 编码；示例 `GET /similar?item_id=85123A&topn=5&method=item2vec&format=json`。推荐带上请求头 `Accept: application/json`。

响应（`format=json` 时）统一规则：

- 成功：HTTP `200`，`Content-Type: application/json`，Body 为单个 JSON 对象，结构随接口不同（见第四节返回字段表）。
- 失败：HTTP `404`，Body 恒为 `{"error": "<原因>"}`（中文可读）。当前实现用 `404` 表示“输入不存在 / 非法”，语义与“资源未找到”一致；建议优化为 `400`（见附录）。
- 未捕获的参数类型异常（如 `topn=abc`）：HTTP `500` —— 属已知边界（见附录改进建议）。

### 编码方式

- 字符集：UTF-8（无 BOM）；中文说明均以 UTF-8 编码传输，`Content-Type` 不再附带 `charset` 之外的特殊编码。
- JSON 序列化：键名 ASCII，展示值含中文；解析时请按 UTF-8 解码。

---

## 三、安全与加密

### 3.1 敏感面说明（如实声明）

本接口为只读推荐查询服务，请求参数仅含 `item_id / user_id / topn / method / ranker / format`，其中：

- `user_id` 为数据集的**内部编号**（4,338 位用户之一），非账号、非口令、非个人可识别信息；
- `item_id`（StockCode）为公开商品目录号。

服务**不接收、不存储、不传输任何口令 / 令牌 / 证件号 / 支付等敏感字段**。因此本版本不存在“明文密码等敏感字段需要传输加密”的实际对象——凡涉及口令加密、密钥协商的能力，均按下文 3.3 以“扩展账号体系时的设计预案”给出，不作为当前已实现功能描述。

### 3.2 传输层安全（机密性 / 完整性 / 身份认证）

生产环境必须启用 HTTPS（TLS 1.2+，推荐 1.3），保证传输机密性与完整性，并由 CA 证书完成服务端身份认证。典型部署（Nginx 终结 TLS）：

- `listen 443 ssl;` 并配置 `ssl_certificate` / `ssl_certificate_key`；
- 拒绝弱加密套件，`ssl_protocols TLSv1.2 TLSv1.3;`
- 回源 `proxy_pass http://127.0.0.1:8000;`（Flask 只监听回环）。
- 可选 `limit_req` 限流、`allow`/`deny` 控制调用方来源。

### 3.3 口令与敏感字段加密 · 密钥协商（设计预案，当前版本未启用账号体系）

若后续扩展“注册 / 登录”账号体系，本组推荐按下列设计实现（当前仓库未包含该能力，验收请勿按“已实现”核对）：

- 口令在传输层：全程 HTTPS。若业务要求在 TLS 之上再做应用层字段加密，采用 **AES-256-GCM**：每条消息使用随机 12 字节 nonce，认证标签 16 字节，杜绝密文重放与篡改。
- 密钥协商（一次握手派生对称会话密钥）：
  1. 客户端携带随机 `client_nonce`（16 字节）与 `client_pub`（若用 ECDH）请求 `POST /auth/key`；
  2. 服务端校验后生成 `server_nonce` + `server_pub`，用双方临时公钥做 **ECDH（P-256）** 派生共享秘密（或用服务端 RSA-2048 公钥加密临时 AES 密钥下发的混合方案）；
  3. 以 KDF（HKDF-SHA256）从共享秘密 + 双 nonce 派生 **AES-256-GCM 会话密钥**；后续字段用其加解密，每条消息 nonce 随机、绝不复用；
  4. 会话结束删除临时密钥——前向保密。
- 口令存储：**绝不明文 / 可逆加密存储**，采用加盐慢哈希（PBKDF2-HMAC-SHA256 ≥ 10 万次迭代，或 bcrypt / Argon2id），每次注册随机盐；登录仅比对哈希。
- 会话凭证：登录成功后由服务端签发 **JWT（HS256 / RS256）**，客户端以 `Authorization: Bearer <token>` 携带；设短有效期并可刷新。

> 注：以上为“若扩展账号体系”时的加密与密钥协商预案，用于满足接口文档对敏感字段加密的规约要求；当前推荐接口本无明文敏感字段。

### 3.4 当前版本已落实的安全措施与边界

- 服务只监听 `127.0.0.1` 回环地址，不对外直接暴露端口；对外须经 HTTPS 反向代理。
- 无 Cookie / Session / 明文凭证写入客户端；输入校验错误均返回明确 `error` 文案（不外泄堆栈）。
- 已知边界（如实）：整型参数未做白名单校验，`topn=abc` 之类会触发 `500`；业务“非法输入”目前以 `404` 表达，建议后续统一改为 `400` + 数值范围校验；未内置鉴权与限流（生产建议由网关 / Nginx 承担）。

---

## 四、接口定义

### 4.1 健康检查 `GET /health`

| 参数 | 类型 | 必填 | 默认 | 业务含义 |
|---|---|---|---|---|
| （无） | — | — | — | 无请求参数 |

正确返回示例（HTTP 200，实际响应原文）：
@@T1@@

| 返回字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 固定 `"ok"` |
| `ready` | boolean | 模型是否加载完毕（冷启动由 `false` → `true`） |

本接口恒成功（200），无业务错误分支；`ready=false` 表示服务仍在加载模型，调用方应轮询等待。

---

### 4.2 商品相似 `GET /similar`

请求 URL 与参数：

| 参数 | 类型 | 必填 | 默认 | 业务含义 |
|---|---|---|---|---|
| `item_id` | string | 是 | — | 查询种子商品码（StockCode，自动转大写），如 `85123A` |
| `topn` | int | 否 | 5 | 返回相似商品个数 |
| `method` | string | 否 | `item2vec` | 相似算法：`item2vec`=同订单成套语义向量（Skip-gram 全量向量）；`itemcf`=同批买家 0/1 画像余弦；**其它取值一律按 `itemcf` 处理** |
| `format` | string | 否 | `html` | `json` → 返回 JSON；否则返回 HTML 演示页 |

正确返回示例（HTTP 200，`/similar?item_id=85123A&topn=3&method=item2vec&format=json` 实际响应原文）：
@@T2@@

返回字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `item_id` | string | 查询种子商品码 |
| `method` | string | 实际使用的算法（`item2vec` / `itemcf`） |
| `query` | object | 种子商品信息：`code` 商品码、`desc` 英文描述、`desc_zh` 中文描述、`category` 类目、`price` 单价£、`buyers` 购买人数 |
| `similar` | array | Top-N 相似商品；每个元素为“query 字段 + `rank`(名次 1 起) + `score`(相似度/余弦, 0~1, 4 位小数)”；**已排除自身**；`desc_zh` 缺失时回退英文 |

错误返回示例（HTTP 404，未传 `item_id` 实际响应原文）：
@@E1@@

其它错误分支：

- 商品不在嵌入词表（出现次数 < 5 的冷门品无向量，仅 `method=item2vec`）→ 404：
  @@E2@@
- 商品不在训练/全量商品（仅 `method=itemcf`）→ 404：`{"error": "商品 X 不在训练/全量商品中"}`
- `topn` 非整数 → 500（已知边界，见附录）。

---

### 4.3 精排推荐 `GET /recommend`

| 参数 | 类型 | 必填 | 默认 | 业务含义 |
|---|---|---|---|---|
| `user_id` | int | 是 | — | 目标用户内部编号，如 `12597`（不存在返回 404） |
| `topn` | int | 否 | 10 | 返回推荐个数 |
| `ranker` | string | 否 | `lr` | 精排模型：`lr`=逻辑回归、`lgb`=LightGBM；其它取值返回 404 |
| `format` | string | 否 | `html` | `json` → JSON；否则 HTML |

服务端流程：三路召回融合为候选池（每用户 ≤ 50）→ 抽取 10 维特征 → `predict_proba` 输出 `P(候选 ∈ 用户下一单)` → 取 Top-N。

正确返回示例（HTTP 200，`/recommend?user_id=12597&topn=3&ranker=lr&format=json` 实际响应原文；`ranker=lgb` 返回结构完全一致，分值略异，见 5.4 截图）：
@@T4@@

返回字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | int | 目标用户编号 |
| `ranker` / `topn` | string / int | 回显请求参数 |
| `user` | object | 用户概览：`n_invoices` 订单数、`n_items` 历史购买去重商品数、`last_basket_zh` 最近一单中文摘要 |
| `recommendations` | array | Top-N 推荐；每个元素：商品信息(见 4.2 的 query 字段) + `rank`(1 起) + `proba`(模型分 0~1) + `routes`(命中的召回路由中文名，多条用 `/` 连接，如 `相似扩展 / 热门`) |

错误返回示例：

- 用户不存在（HTTP 404，`/recommend?user_id=99999999&ranker=lr&format=json` 实际响应原文）：
  @@E3@@
- `ranker` 非法（HTTP 404，`/recommend?user_id=12597&ranker=svm&format=json` 实际响应原文）：
  @@E4@@
- 该用户无可召回候选 → 404：`{"error": "该用户没有可召回候选"}`

---

### 4.4 基线对照 `GET /baseline`

| 参数 | 类型 | 必填 | 默认 | 业务含义 |
|---|---|---|---|---|
| `user_id` | int | 是 | — | 目标用户编号 |
| `topn` | int | 否 | 10 | 每列返回个数 |
| `format` | string | 否 | `html` | `json` → JSON；否则 HTML |

一页并排三条基线（均未经精排），与 `4.3` 对照可见精排增益。

正确返回示例（HTTP 200，`/baseline?user_id=13899&topn=3&format=json` 实际响应原文）：
@@T6@@

返回字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `user_id` | int | 目标用户编号 |
| `user` | object | 用户概览（同 4.3） |
| `last_basket` | array | 该用户最近一单商品码（“事实答案”，仅演示对照用） |
| `columns` | array | 三条基线：`name` ∈ `全局热门 / ItemCF 画像 / 融合召回(未精排)`；`items` 为商品列表，元素含商品信息 + `rank` + `hit`(是否命中 `last_basket`) |

错误返回示例：用户不存在与 `4.3` 相同（404，`{"error": "用户 X 不存在"}`）。

---

### 4.5 错误码表（错误返回速查）

统一错误体：`{"error": "<原因>"}`，均为 `HTTP 404`；HTTP 状态语义：`404`=输入不存在/非法、`200`=成功、`500`=未捕获异常（已知边界）。

| 接口 | 触发条件 | HTTP | `error` 原文 |
|---|---|---|---|
| `/similar` | 未传 `item_id` | 404 | 缺少 item_id 参数，例如 /similar?item_id=85123A&topn=5 |
| `/similar` (item2vec) | `item_id` 不在嵌入词表（出现次数 < 5） | 404 | 商品 X 不在嵌入词表（出现次数 < 5 的冷门品没有向量） |
| `/similar` (itemcf) | `item_id` 不在训练/全量商品 | 404 | 商品 X 不在训练/全量商品中 |
| `/recommend` `/baseline` | `user_id` 不存在 | 404 | 用户 X 不存在（全量共 4,338 位用户） |
| `/recommend` | `ranker` 非 `lr`/`lgb` | 404 | ranker 只能是 lr 或 lgb |
| `/recommend` | 该用户无可召回候选 | 404 | 该用户没有可召回候选 |
| 任意 | `topn`/`user_id` 为非整数 | 500 | （未捕获 ValueError，HTML 错误页）—— 见附录改进建议 |

---

## 五、自测与验收证据（自证可用性）

自测方式：**真实启动本服务**（`http://127.0.0.1:5099`），使用 **Postman 同类工具——浏览器 HTTP 客户端面板**（`outputs/api_doc/api_self_test/live_panel.html`，无头 Chrome 打开；每次打开即向目标接口真实发起一次 HTTP 请求并渲染真实响应）。以下截图均为此工具发出的**真实请求/真实响应记录**，非手工编造；可用文末 `curl` 命令逐条复现。

自测用例与结果：

| 用例 | 接口 | 请求（URL，均加 format=json） | 预期 | 实测 | 结果 |
|---|---|---|---|---|---|
| T1 | /health | /health | 200 `status=ok` | 200 | 通过 |
| T2 | /similar | /similar?item_id=85123A&topn=3&method=item2vec | 200，返回相似商品 | 200 | 通过 |
| T3 | /similar | /similar?item_id=85123A&topn=3&method=itemcf | 200，返回相似商品 | 200 | 通过 |
| T4 | /recommend | /recommend?user_id=12597&topn=3&ranker=lr | 200，Top-3 带 proba | 200 | 通过 |
| T5 | /recommend | /recommend?user_id=12597&topn=3&ranker=lgb | 200，Top-3 带 proba | 200 | 通过 |
| T6 | /baseline | /baseline?user_id=13899&topn=3 | 200，三条基线 | 200 | 通过 |
| E1 | /similar（反例） | /similar?format=json（缺 item_id） | 404 `缺少 item_id 参数` | 404 | 通过 |
| E2 | /similar（反例） | /similar?item_id=ZZZZ9&method=item2vec | 404 `不在嵌入词表` | 404 | 通过 |
| E3 | /recommend（反例） | /recommend?user_id=99999999&ranker=lr | 404 `用户不存在` | 404 | 通过 |
| E4 | /recommend（反例） | /recommend?user_id=12597&ranker=svm | 404 `ranker 只能是 lr 或 lgb` | 404 | 通过 |

### 5.1 健康检查（T1）
@@IMG_T1@@
### 5.2 商品相似（T2 item2vec / T3 itemcf）
@@IMG_T2@@
@@IMG_T3@@
### 5.3 精排推荐（T4 LR / T5 LightGBM）
@@IMG_T4@@
@@IMG_T5@@
### 5.4 基线对照（T6）
@@IMG_T6@@
### 5.5 错误处理（反例 E1–E4）
@@IMG_E1@@
@@IMG_E2@@
@@IMG_E3@@
@@IMG_E4@@

---

## 六、运行、部署与复现

- 数据/模型前置：清洗交易 `data/processed/clean_transactions.csv`、向量索引 `models/vector_index/`、LR/LGB 模型 `models/lr_model.pkl|models/lgb_model.pkl`、演示 ID `models/demo_ids.json`（均已生成）。
- 启动服务：`venv\Scripts\python.exe web\app.py`（默认 `127.0.0.1:5001`；`PORT` 环境变量可覆盖）。
- 命令行自测（与 5.1–5.5 同源同口径）：
  - `curl "http://127.0.0.1:5001/health"`
  - `curl "http://127.0.0.1:5001/similar?item_id=85123A&topn=5&method=item2vec&format=json"`
  - `curl "http://127.0.0.1:5001/recommend?user_id=12597&topn=10&ranker=lr&format=json"`
  - `curl "http://127.0.0.1:5001/baseline?user_id=13899&topn=10&format=json"`
- 重新生成自测截图：`venv\Scripts\python.exe outputs\api_doc\api_self_test\shot_self_test.py`（自动起服务→真实调用→截图→关闭）。
- 演示素材：演示用户 `12597 / 13899`；演示商品 `85123A / 22423`。
- 技术栈：Flask（服务）+ gensim Word2Vec / ItemCF（相似）+ sklearn LogisticRegression、LightGBM（精排）+ Numpy/Pandas（特征与上下文）。

---

## 附录 · 已知边界与改进建议（如实清单）

- 整型参数（`topn`、`user_id`）未做格式校验：传入非整数（如 `topn=abc`）触发未捕获异常 → `500`。改进：统一入参校验中间件，非法参数返回 `400` + `{"error": "参数 topn 必须为整数"}`。
- 业务性“输入不存在 / 非法”当前以 `404` 表达（复用资源未找到语义）。改进：按 `400 参数非法 / 404 资源不存在` 细分。
- `topn` 未设上界、未拦 `0`：`topn=0` 会返回空数组但 HTTP 200。改进：限定 `1 ≤ topn ≤ 100`。
- `/similar` 的 `method` 采取“除 `item2vec` 外一律按 `itemcf`”的宽松处理。改进：枚举校验，非法取值返回 400。
- 演示/课程版本无账号体系，无鉴权与限流；对外生产须叠加 HTTPS 反向代理、网关限流与（若扩展账号体系）3.3 节加密方案。
- 启动加载较重（全量上下文 + 模型，约数十秒）：以 `/health` 的 `ready` 字段作就绪探针。
"""


def img_block(file: str, alt: str, note: str) -> str:
    return f"![{alt}](api_self_test/shots/{file})\n> {note}"


def main() -> None:
    subs = {
        "@@T1@@": cb("T1_health"),
        "@@T2@@": cb("T2_similar_i2v"),
        "@@T4@@": cb("T4_recommend_lr"),
        "@@T6@@": cb("T6_baseline"),
        "@@E1@@": cb("E1_similar_no_item"),
        "@@E2@@": cb("E2_similar_cold"),
        "@@E3@@": cb("E3_recommend_nouser"),
        "@@E4@@": cb("E4_recommend_ranker"),
        "@@IMG_T1@@": img_block("T1_health.png", "健康检查 /health（HTTP 200）", "请求 GET /health → HTTP 200，status=ok、ready=true"),
        "@@IMG_T2@@": img_block("T2_similar_i2v.png", "商品相似 Item2Vec（HTTP 200）", "请求 /similar?item_id=85123A&topn=3&method=item2vec&format=json → HTTP 200，返回 3 条相似商品与相似度"),
        "@@IMG_T3@@": img_block("T3_similar_itemcf.png", "商品相似 ItemCF（HTTP 200）", "请求 /similar?item_id=85123A&topn=3&method=itemcf&format=json → HTTP 200"),
        "@@IMG_T4@@": img_block("T4_recommend_lr.png", "精排推荐 LR（HTTP 200）", "请求 /recommend?user_id=12597&topn=3&ranker=lr&format=json → HTTP 200，Top-3 含 proba 与命中路由"),
        "@@IMG_T5@@": img_block("T5_recommend_lgb.png", "精排推荐 LightGBM（HTTP 200）", "请求 /recommend?user_id=12597&topn=3&ranker=lgb&format=json → HTTP 200，结构同 LR"),
        "@@IMG_T6@@": img_block("T6_baseline.png", "基线对照（HTTP 200）", "请求 /baseline?user_id=13899&topn=3&format=json → HTTP 200，三条基线并列"),
        "@@IMG_E1@@": img_block("E1_similar_no_item.png", "错误用例 E1（HTTP 404）", "反例：缺 item_id → HTTP 404，error=缺少 item_id 参数"),
        "@@IMG_E2@@": img_block("E2_similar_cold.png", "错误用例 E2（HTTP 404）", "反例：冷门品 ZZZZ9 无向量 → HTTP 404，error=不在嵌入词表"),
        "@@IMG_E3@@": img_block("E3_recommend_nouser.png", "错误用例 E3（HTTP 404）", "反例：用户 99999999 不存在 → HTTP 404"),
        "@@IMG_E4@@": img_block("E4_recommend_ranker.png", "错误用例 E4（HTTP 404）", "反例：ranker=svm 非法 → HTTP 404"),
    }
    md = TEMPLATE
    for token, value in subs.items():
        assert token in md, f"模板缺少占位符 {token}"
        md = md.replace(token, value)

    md_path = HERE / "API接口文档素材.md"
    md_path.write_text(md, encoding="utf-8")

    out = PROJECT_ROOT / "outputs" / "API接口文档.docx"
    render(
        md_path, out,
        key_width=[
            ("T6_baseline", 4.2),
            ("T4_recommend_lr", 4.6),
            ("T5_recommend_lgb", 4.6),
            ("T2_similar_i2v", 5.2),
            ("T3_similar_itemcf", 5.2),
            ("T1_health", 5.6),
            ("E1_", 5.7),
            ("E2_", 5.7),
            ("E3_", 5.7),
            ("E4_", 5.7),
        ],
        default_width=5.4,
    )
    print(f"md   -> {md_path}")
    print(f"docx -> {out}  ({out.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
