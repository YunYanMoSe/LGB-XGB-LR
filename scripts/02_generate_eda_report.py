"""阶段二：生成 EDA 可视化 HTML 报告。

读取交易数据（优先 data/processed/clean_transactions.csv，否则 data/raw/transactions.csv），
计算关键指标与各图表数据，渲染出独立的 outputs/eda_report.html（引用 ECharts CDN）。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Windows 控制台默认按 GBK 输出，强制改用 UTF-8，避免中文乱码
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from config.settings import COLUMNS, ENCODINGS, PATHS  # noqa: E402


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_for_eda() -> tuple[pd.DataFrame, str]:
    """读取交易数据（清洗优先），InvoiceNo / StockCode 强制转字符串并解析时间。"""
    clean = PATHS["clean_transactions"]
    if clean.exists():
        df = pd.read_csv(clean, encoding="utf-8")
        source = "data/processed/clean_transactions.csv"
    else:
        df = pd.read_csv(PATHS["raw_transactions"], encoding=ENCODINGS["transactions"])
        source = "data/raw/transactions.csv"

    df[COLUMNS["invoice"]] = df[COLUMNS["invoice"]].astype(str)
    df[COLUMNS["item"]] = df[COLUMNS["item"]].astype(str)
    df[COLUMNS["time"]] = pd.to_datetime(
        df[COLUMNS["time"]], format="%m/%d/%Y %H:%M", errors="coerce"
    )
    return df, source


def load_products_map() -> dict[str, str]:
    """商品主数据 StockCode → 英文描述（同码多描述取第一条）。"""
    prod = pd.read_csv(PATHS["raw_products"], encoding=ENCODINGS["products"])
    prod[COLUMNS["item"]] = prod[COLUMNS["item"]].astype(str)
    prod = prod.drop_duplicates(subset=COLUMNS["item"])
    return dict(zip(prod[COLUMNS["item"]], prod[COLUMNS["desc"]].fillna("")))


# ---------------------------------------------------------------------------
# 指标与图表数据
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, source: str, products_map: dict) -> dict:
    inv, itm = COLUMNS["invoice"], COLUMNS["item"]
    user, country = COLUMNS["user"], COLUMNS["country"]
    qty, price = COLUMNS["qty"], COLUMNS["price"]

    total_rows = int(len(df))
    unique_users = int(df[user].nunique())
    unique_items = int(df[itm].nunique())
    orders = int(df[inv].nunique())
    total_sales = float((df[qty] * df[price]).sum())
    avg_order_value = total_sales / orders if orders else 0.0

    # 各国订单量 Top 10（按发票去重）
    cnt = df.groupby(country)[inv].nunique().sort_values(ascending=False).head(10)
    countries = {
        "names": [str(c) for c in cnt.index],
        "orders": [int(v) for v in cnt.values],
    }

    # 热门商品交互占比 Top 10 + 其余
    vc = df[itm].value_counts().head(10)
    slices = []
    for code, count in vc.items():
        slices.append({
            "name": str(code),
            "value": int(count),
            "desc": products_map.get(str(code), ""),
            "pct": round(float(count) / total_rows * 100, 2),
        })
    top_sum = int(vc.sum())
    slices.append({
        "name": "其余",
        "value": total_rows - top_sum,
        "desc": "Top10 之外的全部商品",
        "pct": round((total_rows - top_sum) / total_rows * 100, 2),
    })

    # 12月1日-9日 每日交易量（按交互行数；结束用次日零点以包含 12-09 全天）
    d0 = pd.Timestamp("2010-12-01")
    d1_incl = pd.Timestamp("2010-12-09")
    sub = df[(df[COLUMNS["time"]] >= d0) & (df[COLUMNS["time"]] < d1_incl + pd.Timedelta(days=1))]
    daily = sub.groupby(sub[COLUMNS["time"]].dt.date).size()
    day_range = pd.date_range(d0, d1_incl)
    dates = [d.strftime("%m/%d") for d in day_range]
    counts = [int(daily.get(d.date(), 0)) for d in day_range]

    # 销量最高的商品 Top 10（按数量合计）
    q = df.groupby(itm)[qty].sum().sort_values(ascending=False).head(10)
    qty_names = [str(c) for c in q.index]
    qty_vals = [int(v) for v in q.values]

    # 单价分布直方图
    edges = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 50, float("inf")]
    labels = ["0~1", "1~2", "2~3", "3~4", "4~5", "5~7",
              "7~10", "10~15", "15~20", "20~50", "50+"]
    bins = pd.cut(df[price], bins=edges, labels=labels, right=True, include_lowest=True)
    hist = bins.value_counts().reindex(labels).fillna(0).astype(int)

    # 每图配套的数据表（可折叠查看）
    tables = {
        "countries": {
            "headers": ["国家", "订单数"],
            "rows": [[n, c] for n, c in zip(countries["names"], countries["orders"])],
        },
        "hot": {
            "headers": ["商品ID", "商品名称", "交互次数", "占比"],
            "rows": [[s["name"], s["desc"] or "-", s["value"], f'{s["pct"]:.2f}%']
                     for s in slices],
        },
        "daily": {
            "headers": ["日期", "交易量（行）"],
            "rows": [[d, c] for d, c in zip(dates, counts)],
        },
        "qty": {
            "headers": ["商品ID", "商品名称", "销量合计"],
            "rows": [[code, products_map.get(code, ""), v]
                     for code, v in zip(qty_names, qty_vals)],
        },
        "hist": {
            "headers": ["单价区间", "交易行数"],
            "rows": [[lbl, c] for lbl, c in zip(labels, hist.tolist())],
        },
    }

    return {
        "source": source,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "currency": "£",
        "metrics": {
            "total_rows": total_rows,
            "unique_users": unique_users,
            "unique_items": unique_items,
            "orders": orders,
            "total_sales": round(total_sales, 2),
            "avg_order_value": round(avg_order_value, 2),
        },
        "countries": countries,
        "hot": {"slices": slices},
        "daily": {"dates": dates, "counts": counts},
        "qty": {"names": qty_names, "values": qty_vals,
                "descs": [products_map.get(c, "") for c in qty_names]},
        "hist": {"labels": labels, "counts": hist.tolist()},
        "tables": tables,
    }


# ---------------------------------------------------------------------------
# HTML 模板（ECharts CDN + 内联主题，含明暗模式）
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>交易数据 EDA 分析报告</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
:root {
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --shadow:0 1px 4px rgba(11,11,11,0.07);
}
* { box-sizing:border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
  -webkit-font-smoothing:antialiased; }
.wrap { max-width:1180px; margin:0 auto; padding:26px 18px 56px; }
header { display:flex; justify-content:space-between; align-items:flex-end; gap:14px; flex-wrap:wrap; }
h1 { font-size:23px; font-weight:720; margin:0; letter-spacing:.2px; }
.sub { color:var(--muted); font-size:12px; margin-top:6px; }
.sub b { color:var(--ink2); font-weight:600; }
.tbtn { border:1px solid var(--border); background:var(--surface); color:var(--muted);
  font-size:12px; border-radius:8px; padding:5px 12px; cursor:pointer;
  font-family:inherit; transition:color .15s, border-color .15s; }
.tbtn:hover { color:var(--ink2); border-color:var(--axis); }

.cards { display:flex; flex-wrap:wrap; gap:14px; justify-content:center; margin:22px 0 26px; }
.card { flex:1 1 172px; max-width:215px; background:var(--surface);
  border:1px solid var(--border); border-radius:14px; padding:18px 16px 14px;
  text-align:center; box-shadow:var(--shadow); }
.card .label { color:var(--muted); font-size:12px; font-weight:600; letter-spacing:.5px; }
.card .value { font-size:22px; font-weight:680; margin-top:9px;
  font-variant-numeric:tabular-nums; letter-spacing:-.3px; }
.card .note { color:var(--ink2); font-size:12px; margin-top:8px; }

.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(460px,1fr)); gap:16px; }
.panel { background:var(--surface); border:1px solid var(--border); border-radius:14px;
  padding:16px 16px 12px; box-shadow:var(--shadow); }
.panel-head { display:flex; justify-content:space-between; align-items:baseline; gap:10px; }
.panel h2 { font-size:15px; font-weight:680; margin:0; }
.p-sub { color:var(--muted); font-size:12px; margin:3px 0 10px; }
.chart { width:100%; height:350px; }
.wide { grid-column:1 / -1; }
.wide .chart { height:320px; }
.tbl { margin-top:12px; overflow-x:auto; }
.tbl table { border-collapse:collapse; font-size:12px; width:100%; }
.tbl th, .tbl td { padding:6px 12px; text-align:right; border-bottom:1px solid var(--grid);
  color:var(--ink2); white-space:nowrap; }
.tbl th { color:var(--muted); font-weight:600; }
.tbl td:first-child, .tbl th:first-child { text-align:left; }
footer { margin-top:34px; color:var(--muted); font-size:12px; text-align:center; line-height:1.8; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <div>
      <h1>交易数据 EDA 分析报告</h1>
      <div class="sub" id="sub-line"></div>
    </div>
    <button class="tbtn" id="themeBtn" title="切换明暗主题">🌙 深色</button>
  </header>

  <section class="cards">
    <div class="card"><div class="label">总行数</div><div class="value" id="m-rows">—</div><div class="note">交互记录总数</div></div>
    <div class="card"><div class="label">唯一用户数</div><div class="value" id="m-users">—</div><div class="note">去重 CustomerID</div></div>
    <div class="card"><div class="label">唯一商品数</div><div class="value" id="m-items">—</div><div class="note">去重 StockCode</div></div>
    <div class="card"><div class="label">总销售额</div><div class="value" id="m-sales">—</div><div class="note" id="note-sales">Quantity × UnitPrice 合计</div></div>
    <div class="card"><div class="label">平均客单价</div><div class="value" id="m-ao">—</div><div class="note" id="note-ao">总销售额 ÷ 订单数</div></div>
  </section>

  <div class="grid">
    <div class="panel">
      <div class="panel-head"><h2>各国订单量 Top 10</h2><button class="tbtn" data-for="countries">数据表</button></div>
      <div class="p-sub">按发票去重统计订单数</div>
      <div id="chart-countries" class="chart"></div>
      <div class="tbl" id="tbl-countries" hidden></div>
    </div>

    <div class="panel">
      <div class="panel-head"><h2>热门商品交互占比 Top 10</h2><button class="tbtn" data-for="hot">数据表</button></div>
      <div class="p-sub">各商品交互行数占全量比例（含其余商品）</div>
      <div id="chart-hotpie" class="chart"></div>
      <div class="tbl" id="tbl-hot" hidden></div>
    </div>

    <div class="panel">
      <div class="panel-head"><h2>12月1日–9日 每日交易量趋势</h2><button class="tbtn" data-for="daily">数据表</button></div>
      <div class="p-sub">2010-12-01 ~ 2010-12-09 · 按交互行数统计</div>
      <div id="chart-daily" class="chart"></div>
      <div class="tbl" id="tbl-daily" hidden></div>
    </div>

    <div class="panel">
      <div class="panel-head"><h2>销量最高的商品 Top 10</h2><button class="tbtn" data-for="qty">数据表</button></div>
      <div class="p-sub">按 Quantity 数量合计排序</div>
      <div id="chart-qty" class="chart"></div>
      <div class="tbl" id="tbl-qty" hidden></div>
    </div>

    <div class="panel wide">
      <div class="panel-head"><h2>单价分布直方图</h2><button class="tbtn" data-for="hist">数据表</button></div>
      <div class="p-sub">UnitPrice 分桶统计交互行数（长尾折叠到 50+）</div>
      <div id="chart-price" class="chart"></div>
      <div class="tbl" id="tbl-hist" hidden></div>
    </div>
  </div>

  <footer>
    数据来自 Online Retail 公开数据集 · 已完成三级清洗（删缺失用户 / 删取消退货 / 删无效价格）<br>
    图表使用 <a href="https://echarts.apache.org/" target="_blank" rel="noopener">Apache ECharts</a> 渲染 · 明暗主题可点击右上角按钮切换
  </footer>
</div>

<script>
const R = __DATA__;

const THEMES = {
  light: {
    page:'#f9f9f7', surface:'#fcfcfb', ink:'#0b0b0b', ink2:'#52514e', muted:'#898781',
    grid:'#e1e0d9', axis:'#c3c2b7', border:'rgba(11,11,11,0.10)',
    shadow:'0 1px 4px rgba(11,11,11,0.07)',
    cat:['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300','#4a3aa7','#e34948'],
    lb1:'#86b6ef', lb2:'#cde2fb', rest:'#898781',
    line:'#2a78d6', area:'rgba(42,120,214,0.16)'
  },
  dark: {
    page:'#0d0d0d', surface:'#1a1a19', ink:'#ffffff', ink2:'#c3c2b7', muted:'#898781',
    grid:'#2c2c2a', axis:'#383835', border:'rgba(255,255,255,0.10)',
    shadow:'0 1px 4px rgba(0,0,0,0.45)',
    cat:['#3987e5','#d95926','#199e70','#c98500','#d55181','#008300','#9085e9','#e66767'],
    lb1:'#86b6ef', lb2:'#cde2fb', rest:'#898781',
    line:'#3987e5', area:'rgba(57,135,229,0.22)'
  }
};

let themeKey = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
let manualTheme = false;
let charts = {};

function applyTheme(t) {
  const s = document.documentElement.style;
  s.setProperty('--page', t.page); s.setProperty('--surface', t.surface);
  s.setProperty('--ink', t.ink); s.setProperty('--ink2', t.ink2);
  s.setProperty('--muted', t.muted); s.setProperty('--grid', t.grid);
  s.setProperty('--axis', t.axis); s.setProperty('--border', t.border);
  s.setProperty('--shadow', t.shadow);
}

function mkTooltip(t) {
  return { backgroundColor:t.surface, borderColor:t.border, borderWidth:1, padding:[8,12],
    textStyle:{ color:t.ink, fontSize:12 }, extraCssText:'border-radius:8px; box-shadow:'+t.shadow+';' };
}
function baseGrid() { return { left:8, right:16, top:32, bottom:8, containLabel:true }; }
function xCat(t, data, rotate) {
  return { type:'category', data:data, boundaryGap:true,
    axisLine:{ lineStyle:{ color:t.axis } }, axisTick:{ show:false },
    axisLabel:{ color:t.muted, fontSize:11, interval:0, rotate:rotate||0 } };
}
function yVal(t, name) {
  return { type:'value', name:name||'', nameTextStyle:{ color:t.muted, fontSize:11 },
    axisLine:{ show:false }, axisTick:{ show:false },
    axisLabel:{ color:t.muted, fontSize:11 }, splitLine:{ lineStyle:{ color:t.grid } } };
}

function barChart(t, el, opt) {
  charts[el] = echarts.init(document.getElementById(el), null, { renderer:'canvas' });
  charts[el].setOption({
    color:[t.cat[0]],
    tooltip: Object.assign(mkTooltip(t), {
      trigger:'axis', axisPointer:{ type:'shadow' },
      formatter: function(ps){
        const p = ps[0];
        const d = opt.descs ? opt.descs[p.dataIndex] : '';
        return '<b>'+p.axisValueLabel+'</b>'+(d?' · '+d:'')+'<br/>'+opt.name+': '+p.value.toLocaleString('en-US');
      }
    }),
    grid: baseGrid(),
    xAxis: Object.assign(xCat(t, opt.names, opt.xRot||0), { boundaryGap:true }),
    yAxis: yVal(t, opt.yName),
    series:[{
      name:opt.name, type:'bar', data:opt.values, barMaxWidth:30,
      itemStyle:{ color:t.cat[0], borderRadius:[4,4,0,0] },
      label:{ show:true, position:'top', color:t.ink2, fontSize:11,
        formatter:function(p){ return p.value.toLocaleString('en-US'); } }
    }]
  });
}

function lineChart(t, el, opt) {
  charts[el] = echarts.init(document.getElementById(el), null, { renderer:'canvas' });
  charts[el].setOption({
    color:[t.line],
    tooltip: Object.assign(mkTooltip(t), {
      trigger:'axis',
      axisPointer:{ type:'cross', crossStyle:{ color:t.axis }, label:{ backgroundColor:t.axis, color:t.surface } }
    }),
    grid: baseGrid(),
    xAxis: Object.assign(xCat(t, opt.names), { boundaryGap:false }),
    yAxis: yVal(t, opt.yName),
    series:[{
      name:opt.name, type:'line', data:opt.values, symbolSize:8,
      lineStyle:{ width:2, color:t.line },
      itemStyle:{ color:t.line, borderColor:t.surface, borderWidth:1 },
      areaStyle:{ color:{ type:'linear', x:0, y:0, x2:0, y2:1,
        colorStops:[{ offset:0, color:t.area }, { offset:1, color:'transparent' }] } }
    }]
  });
}

function pieChart(t, el, slices) {
  const colors = t.cat.concat([t.lb1, t.lb2, t.rest]);
  charts[el] = echarts.init(document.getElementById(el), null, { renderer:'canvas' });
  charts[el].setOption({
    tooltip: Object.assign(mkTooltip(t), {
      trigger:'item',
      formatter: function(p){
        const d = p.data;
        return (d.desc?'<b>'+d.desc+'</b><br/>':'')+'<b>'+p.name+'</b><br/>交互次数: '+
          p.value.toLocaleString('en-US')+'<br/>占比: '+(typeof d.pct==='number'?d.pct.toFixed(2):'0.00')+'%';
      }
    }),
    legend:{ type:'scroll', orient:'vertical', right:0, top:'middle',
      textStyle:{ color:t.ink2, fontSize:11 }, itemWidth:10, itemHeight:10, icon:'circle' },
    series:[{
      type:'pie', radius:['34%','70%'], center:['40%','50%'],
      data:slices, color:colors,
      label:{ color:t.ink2, fontSize:11, formatter:'{b}\\n{d}%' },
      labelLine:{ length:12, length2:8, lineStyle:{ color:t.axis, width:1 } },
      itemStyle:{ borderColor:t.surface, borderWidth:2 },
      emphasis:{ scaleSize:5, label:{ fontSize:12 } }
    }]
  });
}

function renderTable(id) {
  const tab = R.tables[id];
  let html = '<table><thead><tr>';
  html += tab.headers.map(function(h){ return '<th>'+h+'</th>'; }).join('');
  html += '</tr></thead><tbody>';
  html += tab.rows.map(function(r){
    return '<tr>'+r.map(function(c){ return '<td>'+c+'</td>'; }).join('')+'</tr>';
  }).join('');
  html += '</tbody></table>';
  const box = document.getElementById('tbl-'+id);
  box.innerHTML = html;
  box.hidden = !box.hidden;
}

function renderAll() {
  const t = THEMES[themeKey];
  applyTheme(t);
  Object.keys(charts).forEach(function(k){ try{ charts[k].dispose(); }catch(e){} });
  charts = {};

  const m = R.metrics;
  document.getElementById('m-rows').textContent = m.total_rows.toLocaleString('en-US');
  document.getElementById('m-users').textContent = m.unique_users.toLocaleString('en-US');
  document.getElementById('m-items').textContent = m.unique_items.toLocaleString('en-US');
  document.getElementById('m-sales').textContent = R.currency + Math.round(m.total_sales).toLocaleString('en-US');
  document.getElementById('m-ao').textContent = R.currency + m.avg_order_value.toFixed(2);
  document.getElementById('note-sales').textContent = 'Quantity × UnitPrice 合计（'+R.currency+'）';
  document.getElementById('note-ao').textContent = '总销售额 ÷ 订单数（'+m.orders.toLocaleString('en-US')+' 单）';
  document.getElementById('sub-line').innerHTML =
    '数据源: <b>'+R.source+'</b> · 生成时间: '+R.generated_at+' · 共 '+m.total_rows.toLocaleString('en-US')+' 行 / '+m.orders.toLocaleString('en-US')+' 张订单';

  barChart(t, 'chart-countries', { names:R.countries.names, values:R.countries.orders, name:'订单数', xRot:28 });
  pieChart(t, 'chart-hotpie', R.hot.slices);
  lineChart(t, 'chart-daily', { names:R.daily.dates, values:R.daily.counts, name:'交易量', yName:'交易量（行）' });
  barChart(t, 'chart-qty', { names:R.qty.names, values:R.qty.values, name:'销量合计', descs:R.qty.descs });
  barChart(t, 'chart-price', { names:R.hist.labels, values:R.hist.counts, name:'交易行数', yName:'行数' });

  document.getElementById('themeBtn').textContent = themeKey==='dark' ? '☀️ 浅色' : '🌙 深色';
}

document.addEventListener('DOMContentLoaded', function(){
  document.getElementById('themeBtn').addEventListener('click', function(){
    themeKey = themeKey==='dark' ? 'light' : 'dark';
    manualTheme = true;
    renderAll();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e){
    if (!manualTheme) { themeKey = e.matches ? 'dark' : 'light'; renderAll(); }
  });
  window.addEventListener('resize', function(){
    Object.keys(charts).forEach(function(k){ charts[k].resize(); });
  });
  document.querySelectorAll('.tbtn[data-for]').forEach(function(btn){
    btn.addEventListener('click', function(){ renderTable(btn.getAttribute('data-for')); });
  });
  renderAll();
});
</script>
</body>
</html>
"""


def main() -> None:
    df, source = load_for_eda()
    products_map = load_products_map()
    report = build_report(df, source, products_map)

    out_path = PATHS["outputs_dir"] / "eda_report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data_json = json.dumps(report, ensure_ascii=False, allow_nan=False)
    html = HTML_TEMPLATE.replace("__DATA__", data_json)
    out_path.write_text(html, encoding="utf-8")

    print(f"[EDA] 报告已生成: {out_path}")


if __name__ == "__main__":
    main()
