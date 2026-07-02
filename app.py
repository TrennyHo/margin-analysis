#!/usr/bin/env python3
"""
融資分析 Web App
Usage: python3 app.py
瀏覽器開 http://localhost:5566
"""

import io, os, time, base64, requests, warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from flask import Flask, render_template, request, jsonify

warnings.filterwarnings("ignore", message="Unverified HTTPS request")

# ── 載入打包字型（確保 Render 等 Linux 環境顯示中文）──────
import matplotlib.font_manager as _fm
_FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "wqy-microhei.ttc")
if os.path.exists(_FONT_PATH):
    _fm.fontManager.addfont(_FONT_PATH)

app = Flask(__name__)

MONTHS  = 5
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
MARKET_KEYWORDS = {"大盤","加權","twii","taiex","市場","大盤指數","加權指數"}

# ── 股票清單（名稱 ↔ 代碼）─────────────────────────────
_stock_map: dict[str, str] = {}

def load_stock_map():
    global _stock_map
    if _stock_map:
        return
    try:
        r = requests.get(
            "https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN"
            f"?date={datetime.now().strftime('%Y%m%d')}&selectType=ALL",
            headers=HEADERS, timeout=12,
        )
        for item in r.json():
            if isinstance(item, dict):
                _stock_map[item["股票代號"]] = item["股票名稱"]
    except Exception:
        pass

def search_stock(query: str) -> tuple[str, str] | None:
    load_stock_map()
    q = query.strip()
    if q in _stock_map:
        return q, _stock_map[q]
    for code, name in _stock_map.items():
        if q in name:
            return code, name
    return None

# ── 工具 ──────────────────────────────────────────────
def roc_to_date(s):
    p = s.strip().split("/")
    return datetime(int(p[0]) + 1911, int(p[1]), int(p[2]))

def prev_months(n):
    now = datetime.now()
    result, y, m = [], now.year, now.month
    for _ in range(n):
        result.append(f"{y}{m:02d}01")
        m -= 1
        if m == 0:
            m, y = 12, y - 1
    return result

# ── 個股股價 ──────────────────────────────────────────
def fetch_price(stock_no):
    rows = []
    for ds in prev_months(MONTHS):
        url = (f"https://www.twse.com.tw/rwd/zh/afterTrading/STOCK_DAY"
               f"?date={ds}&stockNo={stock_no}&response=json")
        try:
            d = requests.get(url, headers=HEADERS, timeout=12).json()
            if d.get("stat") == "OK":
                rows.extend(d.get("data", []))
        except Exception:
            pass
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    COLS = ["日期","成交股數","成交金額","開盤價","最高價","最低價","收盤價","漲跌價差","成交筆數"]
    df = pd.DataFrame([r[:9] for r in rows], columns=COLS)
    df["date"] = df["日期"].apply(roc_to_date)
    for c in ["開盤價","最高價","最低價","收盤價"]:
        df[c] = pd.to_numeric(df[c].str.replace(",",""), errors="coerce")
    df = df.dropna(subset=["收盤價"]).sort_values("date").reset_index(drop=True)
    df["MA5"]  = df["收盤價"].rolling(5).mean()
    df["MA20"] = df["收盤價"].rolling(20).mean()
    return df

# ── 加權指數（大盤）──────────────────────────────────
def fetch_taiex():
    rows = []
    for ds in prev_months(MONTHS):
        url = (f"https://www.twse.com.tw/rwd/zh/afterTrading/FMTQIK"
               f"?date={ds}&response=json")
        try:
            d = requests.get(url, headers=HEADERS, timeout=12).json()
            if d.get("stat") == "OK":
                rows.extend(d.get("data", []))
        except Exception:
            pass
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame()
    # fields: 日期, 成交股數, 成交金額, 成交筆數, 發行量加權股價指數
    df = pd.DataFrame(rows)
    df.columns = df.columns if len(df.columns) > 1 else ["日期","成交股數","成交金額","成交筆數","指數"]
    df = df.iloc[:, [0, 4]]
    df.columns = ["日期", "指數"]
    df["date"] = df["日期"].apply(roc_to_date)
    df["指數"] = pd.to_numeric(df["指數"].str.replace(",",""), errors="coerce")
    df = df.dropna(subset=["指數"]).sort_values("date").reset_index(drop=True)
    df["MA5"]  = df["指數"].rolling(5).mean()
    df["MA20"] = df["指數"].rolling(20).mean()
    return df

# ── 個股融資（TWSE openAPI，全球可存取）────────────────
def fetch_margin(stock_no):
    from concurrent.futures import ThreadPoolExecutor, as_completed as _ac

    # 先取股價資料（同時獲得交易日清單）
    df_price = fetch_price(stock_no)
    if df_price.empty:
        return pd.DataFrame()
    trade_dates = df_price["date"].tolist()

    def _n(s):
        return pd.to_numeric(str(s).replace(",", ""), errors="coerce")

    def _fetch_day(dt):
        ymd = dt.strftime("%Y%m%d")
        url = (f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
               f"?response=json&date={ymd}&selectType=STOCK")
        try:
            r = requests.get(url, headers=HEADERS, timeout=12)
            d = r.json()
            tables = d.get("tables", [])
            data = tables[1].get("data", []) if len(tables) > 1 else []
            for row in data:
                if row[0] == stock_no:
                    # row: [代號,名稱,融資買進,融資賣出,現金償還,融資前日餘額,融資今日餘額,融資限額,
                    #        融券買進,融券賣出,現券償還,融券前日餘額,融券今日餘額,融券限額,資券互抵,註記]
                    return dt, (_n(row[6]), _n(row[7]), _n(row[12]), _n(row[13]))
        except Exception:
            pass
        return dt, None

    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        fts = {ex.submit(_fetch_day, dt): dt for dt in trade_dates}
        for f in _ac(fts):
            dt, vals = f.result()
            if vals:
                results[dt] = vals

    if not results:
        return pd.DataFrame()

    rows = [{"date": dt,
             "融資今日餘額": v[0],
             "融資限額":    v[1],
             "融券今日餘額": v[2],
             "融券限額":    v[3],
             } for dt, v in sorted(results.items())]

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["融資日變化"] = df["融資今日餘額"].diff()
    df["融券日變化"] = df["融券今日餘額"].diff()
    df["融資比率"]  = (df["融資今日餘額"] / df["融資限額"] * 100).round(2)
    df["券資比"]    = (df["融券今日餘額"] / df["融資今日餘額"] * 100).round(2)

    # 合併股價（收盤價 / MA5 / MA20）
    df = df.merge(df_price[["date", "收盤價", "MA5", "MA20"]], on="date", how="left")
    return df.dropna(subset=["融資今日餘額"]).reset_index(drop=True)

# ── 大盤融資（TWSE 官方市場總計，全球可存取）─────────
def fetch_market_margin(trade_dates):
    from concurrent.futures import ThreadPoolExecutor, as_completed as _ac

    def _fetch_day(dt):
        ymd = dt.strftime("%Y%m%d")
        url = (f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN"
               f"?response=json&date={ymd}")
        try:
            d = requests.get(url, headers=HEADERS, timeout=12).json()
            if d.get("stat") == "OK" and d.get("tables"):
                data = d["tables"][0].get("data", [])
                if len(data) >= 2:
                    # fields: 項目, 買進, 賣出, 現金(券)償還, 前日餘額, 今日餘額
                    margin = pd.to_numeric(str(data[0][5]).replace(",", ""), errors="coerce")
                    short  = pd.to_numeric(str(data[1][5]).replace(",", ""), errors="coerce")
                    if pd.notna(margin):
                        return dt, margin, short
        except Exception:
            pass
        return dt, None, None

    results = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        fts = {ex.submit(_fetch_day, dt): dt for dt in trade_dates}
        for f in _ac(fts):
            dt, margin, short = f.result()
            if margin is not None:
                results[dt] = (margin, short or 0)

    if not results:
        return pd.DataFrame()

    rows = [{"date": dt, "融資餘額": v[0], "融券餘額": v[1]}
            for dt, v in sorted(results.items())]
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["融資日變化"] = df["融資餘額"].diff()
    df["融資萬張"]  = (df["融資餘額"] / 1e4).round(2)
    return df

# ── 畫圖共用：緊縮 Y 軸（讓變化看得見）─────────────
def tight_ylim(ax, series, margin=0.15):
    mn, mx = series.min(), series.max()
    pad = (mx - mn) * margin if mx > mn else abs(mx) * 0.05 or 1
    ax.set_ylim(mn - pad, mx + pad)

def tight_ylim_zero(ax, series, margin=0.15):
    """日變化圖：保留 0 軸，但上下都緊縮"""
    mn, mx = min(series.min(), 0), max(series.max(), 0)
    pad = max((mx - mn) * margin, abs(mx) * 0.05, 1)
    ax.set_ylim(mn - pad, mx + pad)

# ── 共用畫布設定（可指定格數）──────────────────────────
def _base_fig(title, n_panels=4, height_ratios=None):
    plt.rcParams["font.family"] = ["WenQuanYi Micro Hei","Heiti TC","PingFang HK","STHeiti","Noto Sans CJK TC","Noto Sans CJK SC","WenQuanYi Zen Hei","sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    BG, GRID = "#0e1117", "#2a2d3e"
    if height_ratios is None:
        height_ratios = [3] + [1.5] * (n_panels - 1)
    fig_h = 12 + (n_panels - 4) * 2
    fig = plt.figure(figsize=(15, fig_h), facecolor=BG)
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.97)
    gs = fig.add_gridspec(n_panels, 1, hspace=0.06, height_ratios=height_ratios,
                          left=0.08, right=0.95, top=0.94, bottom=0.06)
    axes = [fig.add_subplot(gs[i]) for i in range(n_panels)]
    for i in range(1, n_panels):
        axes[i].sharex(axes[0])
    for ax in axes:
        ax.set_facecolor(BG)
        ax.grid(True, color=GRID, lw=0.5)
        ax.tick_params(colors="gray", labelsize=10)
        ax.yaxis.set_tick_params(labelcolor="white", labelsize=10)
        for sp in ax.spines.values(): sp.set_color(GRID)
    for i in range(n_panels - 1):
        plt.setp(axes[i].xaxis.get_majorticklabels(), visible=False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha="right", color="gray", fontsize=10)
    return fig, axes, BG, GRID

LEG_KW = dict(loc="upper left", fontsize=11, facecolor="#1a1d2e", labelcolor="white", framealpha=0.85, edgecolor="#4a4d5e")

# ── 個股圖 ────────────────────────────────────────────
def make_stock_chart(stock_no, stock_name, df_m):
    df = df_m.dropna(subset=["融資今日餘額"]).copy()
    if df.empty:
        return None

    fig, axes_list, BG, GRID = _base_fig(
        f"融資分析  {stock_no} {stock_name}  （近 {len(df)} 個交易日，資料來源 CMoney）",
        n_panels=5
    )
    ax1, ax2, ax3, ax4, ax5 = axes_list
    dates = df["date"]

    # ── Panel 1：股價（CMoney，與融資同樣 120 天）──
    ax1.plot(dates, df["收盤價"], color="#4fc3f7", lw=1.2, label="收盤價")
    ax1.plot(dates, df["MA5"],   color="#ffb74d", lw=0.9, ls="--", label="MA5")
    ax1.plot(dates, df["MA20"],  color="#81c784", lw=0.9, ls="--", label="MA20")
    ax1.set_ylabel("股價", color="white", fontsize=10)
    ax1.legend(**LEG_KW)

    # ── Panel 2：融資餘額 ──
    ax2.plot(dates, df["融資今日餘額"], color="#42a5f5", lw=1.8, label="融資餘額（張）")
    tight_ylim(ax2, df["融資今日餘額"], margin=0.2)
    ax2.set_ylabel("融資餘額（張）", color="#42a5f5", fontsize=11)
    ax2.yaxis.set_tick_params(labelcolor="#42a5f5")
    ax2.legend(**LEG_KW)

    # ── Panel 3：融資比率 ──
    ratio = df["融資比率"].dropna()
    if not ratio.empty:
        ax3.plot(dates.iloc[ratio.index], ratio, color="#ff7043", lw=1.8, label="融資比率 %")
        tight_ylim(ax3, ratio, margin=0.2)
        if ratio.max() > 15:
            ax3.axhline(y=20, color="#ff7043", lw=0.8, ls=":", alpha=0.55)
    ax3.set_ylabel("融資比率 %", color="#ff7043", fontsize=11)
    ax3.yaxis.set_tick_params(labelcolor="#ff7043")
    ax3.legend(**LEG_KW)

    # ── Panel 4：融資日變化 ──
    chg = df["融資日變化"].fillna(0)
    clrs = ["#ef5350" if v < 0 else "#26a69a" for v in chg]
    ax4.bar(dates, chg, color=clrs, width=1.5, label="融資日變化（張）")
    ax4.axhline(y=0, color="white", lw=0.8, alpha=0.5)
    tight_ylim_zero(ax4, chg)
    ax4.set_ylabel("日變化（張）", color="white", fontsize=11)
    ax4.legend(**LEG_KW)

    # ── Panel 5：融券餘額 + 券資比 ──
    short = df["融券今日餘額"].fillna(0)
    ax5.plot(dates, short, color="#ce93d8", lw=1.8, label="融券餘額（張）")
    tight_ylim(ax5, short, margin=0.2)
    ax5.set_ylabel("融券餘額（張）", color="#ce93d8", fontsize=11)
    ax5.yaxis.set_tick_params(labelcolor="#ce93d8")
    qr = df["券資比"].dropna()
    if not qr.empty:
        ax5r = ax5.twinx()
        ax5r.plot(dates.iloc[qr.index], qr, color="#fff176", lw=1.2, ls="--", alpha=0.8, label="券資比 %")
        ax5r.set_ylabel("券資比 %", color="#fff176", fontsize=11)
        ax5r.yaxis.set_tick_params(labelcolor="#fff176")
        ax5r.set_facecolor("#0e1117")
        for sp in ax5r.spines.values(): sp.set_color("#2a2d3e")
        h1, l1 = ax5.get_legend_handles_labels()
        h2, l2 = ax5r.get_legend_handles_labels()
        ax5.legend(h1 + h2, l1 + l2, **LEG_KW)
    else:
        ax5.legend(**LEG_KW)

    last = df.iloc[-1]
    rate_str  = f"{last['融資比率']:.2f}%"   if pd.notna(last.get("融資比率")) else "—"
    qr_str    = f"{last['券資比']:.2f}%"     if pd.notna(last.get("券資比"))   else "—"
    short_str = f"{last['融券今日餘額']:,.0f}" if pd.notna(last.get("融券今日餘額")) else "—"
    close_val = last['收盤價'] if pd.notna(last.get('收盤價')) else 0
    fig.text(0.5, 0.002,
             f"最新：{last['date'].strftime('%Y/%m/%d')}　收盤 {close_val:.1f}　"
             f"融資 {last['融資今日餘額']:,.0f} 張（{rate_str}）　"
             f"融券 {short_str} 張　券資比 {qr_str}",
             ha="center", color="#aaaaaa", fontsize=11)
    return _fig_to_b64(fig, BG)

# ── 大盤圖 ────────────────────────────────────────────
def make_market_chart(df_idx, df_m):
    df = df_m.dropna(subset=["融資餘額"]).copy()
    if df.empty:
        return None

    fig, axes_m, BG, GRID = _base_fig(
        f"大盤融資分析  加權指數 × 全市場融資總計  （近 {len(df)} 個交易日）"
    )
    ax1, ax2, ax3, ax4 = axes_m
    dates = df["date"]

    # ── Panel 1：加權指數 ──
    ax1.plot(df_idx["date"], df_idx["指數"], color="#4fc3f7", lw=1.2, label="加權指數")
    ax1.plot(df_idx["date"], df_idx["MA5"],  color="#ffb74d", lw=0.9, ls="--", label="MA5")
    ax1.plot(df_idx["date"], df_idx["MA20"], color="#81c784", lw=0.9, ls="--", label="MA20")
    ax1.set_ylabel("加權指數", color="white", fontsize=10)
    ax1.legend(**LEG_KW)

    # ── Panel 2：成份股融資餘額加總 ──
    ax2.plot(dates, df["融資萬張"], color="#42a5f5", lw=1.8, label="融資餘額（萬張，全市場）")
    tight_ylim(ax2, df["融資萬張"], margin=0.2)
    ax2.set_ylabel("融資餘額（萬張）", color="#42a5f5", fontsize=11)
    ax2.yaxis.set_tick_params(labelcolor="#42a5f5")
    ax2.legend(**LEG_KW)

    # ── Panel 3：融券餘額 ──
    short_wan = df["融券餘額"] / 1e4
    ax3.plot(dates, short_wan, color="#ff7043", lw=1.8, label="融券餘額（萬張）")
    tight_ylim(ax3, short_wan, margin=0.2)
    ax3.set_ylabel("融券餘額（萬張）", color="#ff7043", fontsize=11)
    ax3.yaxis.set_tick_params(labelcolor="#ff7043")
    ax3.legend(**LEG_KW)

    # ── Panel 4：融資日變化 ──
    chg = df["融資日變化"].fillna(0)
    clrs = ["#ef5350" if v < 0 else "#26a69a" for v in chg]
    ax4.bar(dates, chg, color=clrs, width=1.5, label="融資日變化（張）")
    ax4.axhline(y=0, color="white", lw=0.8, alpha=0.5)
    tight_ylim_zero(ax4, chg)
    ax4.set_ylabel("日變化（張）", color="white", fontsize=11)
    ax4.legend(**LEG_KW)

    last = df.iloc[-1]
    idx_last = df_idx.iloc[-1]
    chg_last = last.get("融資日變化", float("nan"))
    footer = (f"最新：{last['date'].strftime('%Y/%m/%d')}　"
              f"加權 {idx_last['指數']:,.2f}　"
              f"全市場融資合計 {last['融資萬張']:.2f} 萬張　"
              f"日變化 {chg_last:+,.0f} 張" if pd.notna(chg_last) else "")
    fig.text(0.5, 0.002, footer, ha="center", color="#aaaaaa", fontsize=11)
    return _fig_to_b64(fig, BG)

def _fig_to_b64(fig, bg):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── Flask 路由 ─────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    query = request.json.get("query", "").strip()
    if not query:
        return jsonify(error="請輸入股票代碼或名稱"), 400

    # 大盤模式
    if query.lower() in MARKET_KEYWORDS:
        df_idx = fetch_taiex()
        if df_idx.empty:
            return jsonify(error="無法取得加權指數資料"), 404
        df_m = fetch_market_margin(df_idx["date"].tolist())
        if df_m.empty:
            return jsonify(error="無法取得大盤融資資料"), 404
        img = make_market_chart(df_idx, df_m)
        if img is None:
            return jsonify(error="圖表產生失敗"), 500
        last = df_m.iloc[-1]
        chg = last.get("融資日變化", float("nan"))
        stats = {
            "date":        last["date"].strftime("%Y/%m/%d"),
            "close":       f"{df_idx.iloc[-1]['指數']:,.2f}",
            "margin_bal":  f"{last['融資萬張']:.1f} 萬",
            "margin_lim":  "全市場總計",
            "margin_rate": "—",
            "margin_chg":  f"{chg:+,.0f}" if pd.notna(chg) else "—",
            "short_bal":   f"{last['融券餘額']/1e4:.2f} 萬",
            "risk":        "—",
        }
        return jsonify(chart=img, stats=stats, name="全市場融資總計", code="大盤")

    # 個股模式
    if query.isdigit():
        stock_no = query
        load_stock_map()
        stock_name = _stock_map.get(stock_no, "")
    else:
        result = search_stock(query)
        if not result:
            return jsonify(error=f"找不到「{query}」，請確認名稱或改用代碼"), 404
        stock_no, stock_name = result

    df_margin = fetch_margin(stock_no)
    if df_margin.empty:
        return jsonify(error=f"{stock_no} 無融資資料（ETF 或無融資資格）"), 404

    img = make_stock_chart(stock_no, stock_name, df_margin)
    if img is None:
        return jsonify(error="圖表產生失敗"), 500

    last  = df_margin.iloc[-1]
    rate  = last.get("融資比率")
    lim   = last.get("融資限額")
    close = last.get("收盤價", 0)
    stats = {
        "date":        last["date"].strftime("%Y/%m/%d"),
        "close":       f"{close:.1f}" if pd.notna(close) else "—",
        "margin_bal":  f"{last['融資今日餘額']:,.0f}",
        "margin_lim":  f"{lim:,.0f}" if pd.notna(lim) else "—",
        "margin_rate": f"{rate:.2f}"  if pd.notna(rate) else "—",
        "margin_chg":  f"{last['融資日變化']:+,.0f}" if pd.notna(last.get("融資日變化")) else "—",
        "short_bal":   f"{last['融券今日餘額']:,.0f}",
        "risk":        "高" if pd.notna(rate) and rate > 20 else "中" if pd.notna(rate) and rate > 10 else "低",
    }
    return jsonify(chart=img, stats=stats, name=stock_name, code=stock_no)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5566))
    print(f"\n  融資分析 App 啟動中...")
    print(f"  請開瀏覽器進入 → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
