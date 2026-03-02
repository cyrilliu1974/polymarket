import streamlit as st
import subprocess
import re
import os
import requests
import numpy as np
from itertools import combinations
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Polymarket Copula Bot", layout="wide")
st.title("📊 Polymarket Copula Bot")
st.markdown("輸入 2～5 個市場 Slug，模擬聯合機率分佈")

# ==================== Slug 輸入欄位 ====================
st.subheader("🎯 市場 Slug 輸入")
st.caption("每個欄位填入一個 Polymarket 市場 Slug（必填 2 個，最多 5 個）")

default_slugs = [
    "will-china-invade-taiwan-before-2027",
    "china-x-taiwan-military-clash-before-2027",
    "", "", ""
]

slug_inputs = []
for i in range(5):
    val = st.text_input(
        f"Slug {i+1} {'（必填）' if i < 2 else '（選填）'}",
        value=default_slugs[i],
        placeholder="e.g. will-bitcoin-hit-100k-in-march",
        key=f"slug_{i}"
    )
    slug_inputs.append(val.strip())

# ==================== 模擬設定 ====================
st.subheader("⚙️ 模擬設定")
num_sim = st.slider(
    "模擬次數",
    min_value=10000,
    max_value=500000,
    value=150000,
    step=10000,
    help="次數越高結果越精確，但執行時間越長"
)
st.caption(f"目前設定：{num_sim:,} 次")

# ==================== 相關性設定 ====================
st.subheader("🔗 相關性設定")
st.caption("相關性代表「兩個事件同時發生」的傾向強度。0 = 完全獨立，1 = 幾乎同進退")

global_corr = st.slider(
    "全域相關性強度（套用到所有市場配對）",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    help="例如台海兩個市場高度相關可設 0.9，比特幣與台海關聯較低可設 0.2"
)

# 過濾已填入的 slug，用於顯示進階設定
active_slugs = [s for s in slug_inputs if s.strip()]

# 進階：每對市場獨立設定
pair_corr = {}
if len(active_slugs) >= 2:
    with st.expander("⚙️ 進階：自訂每對市場的相關性（展開後可覆蓋全域設定）"):
        st.caption("每一對市場可以設定不同的相關性。同主題的兩個市場可設高一點，不同主題則設低一點。實際送出時會取所有配對的平均值。")
        pairs = list(combinations(range(len(active_slugs)), 2))
        for (i, j) in pairs:
            label = f"Slug {i+1} ↔ Slug {j+1}：`{active_slugs[i]}` vs `{active_slugs[j]}`"
            val = st.slider(
                label,
                min_value=0.0,
                max_value=1.0,
                value=global_corr,
                step=0.05,
                key=f"corr_{i}_{j}"
            )
            pair_corr[(i, j)] = val

# ==================== 執行按鈕 ====================
st.divider()
run_btn = st.button("🚀 執行模擬 + 自動解析", type="primary", use_container_width=True)

if run_btn:
    slugs = [s for s in slug_inputs if s]

    invalid_slugs = [s for s in slugs if not re.match(r'^[\w-]+$', s)]
    if invalid_slugs:
        st.error(f"❌ 以下 Slug 格式錯誤（只允許英文、數字、連字號）：{invalid_slugs}")
        st.stop()

    if len(slugs) < 2:
        st.error("❌ 請至少填入 2 個有效的 Slug")
        st.stop()

    # 計算實際使用的 corr_strength
    if pair_corr:
        effective_corr = round(float(np.mean(list(pair_corr.values()))), 2)
    else:
        effective_corr = global_corr

    markets_str = ",".join(slugs)
    args_list = [
        "--mode", "correlated_demo",
        "--markets", markets_str,
        "--copula", "t",
        "--num_sim", str(num_sim),
        "--corr_strength", str(effective_corr)
    ]

    st.info(f"📋 執行市場：{' | '.join(slugs)}　｜　有效相關性：{effective_corr}")

    current_env = os.environ.copy()
    current_env["PYTHONIOENCODING"] = "utf-8"

    with st.spinner(f"正在模擬 {num_sim:,} 次，請稍候..."):
        result = subprocess.run(
            ["python", "polymarket_simulation_bot.py"] + args_list,
            capture_output=True, encoding="utf-8", errors="replace",
            env=current_env, timeout=600
        )

    output = result.stdout
    error_log = result.stderr

    # ==================== 執行失敗處理 ====================
    if result.returncode != 0:
        st.error("❌ Bot 執行失敗")
        fail_matches = re.findall(r"❌ \[關鍵錯誤\](.*?)(?=❌|$)", output, re.DOTALL)
        if fail_matches:
            for msg in fail_matches:
                st.warning(f"**診斷：** {msg.strip()}")
        else:
            st.warning("請檢查下方完整日誌")
        with st.expander("🔍 查看完整錯誤日誌"):
            st.code(output + "\n" + error_log)
        st.stop()

    # ==================== 解析結果 ====================
    def safe_float(pattern, text):
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1))
            except:
                return None
        return None

    p_indep = safe_float(r"市場獨立掃全部機率:\s*([\d.]+)", output)
    p_sweep = safe_float(r"模擬掃全部機率:\s*([\d.]+)", output)
    p_lose  = safe_float(r"模擬全輸機率:\s*([\d.]+)", output)
    market_probs = re.findall(r"✅ 抓到 ([\w-]+)：機率 ([\d.]+)%", output)

    if p_indep is not None and p_sweep is not None:
        edge = p_sweep - p_indep
        multiplier = p_sweep / p_indep if p_indep > 0 else float('inf')

        st.success("✅ 模擬完成，以下是解析結果")

        # ── 各市場個別機率 ──
        if market_probs:
            st.subheader("📌 各市場現況")
            prob_cols = st.columns(len(market_probs))
            for idx, (slug, prob_str) in enumerate(market_probs):
                with prob_cols[idx]:
                    st.metric(label=slug, value=f"{prob_str}%")

        st.divider()

        # ── 聯合機率主結果 ──
        st.subheader("📊 聯合機率模擬結果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("獨立相乘機率", f"{p_indep:.4f}",
                      help="假設各市場完全獨立時的聯合機率（市場目前的定價邏輯）")
        with col2:
            delta_color = "normal" if edge >= 0 else "inverse"
            st.metric("T-Copula 聯合掃全部", f"{p_sweep:.4f}",
                      delta=f"{edge:+.4f} edge", delta_color=delta_color,
                      help="考慮尾部相關性後的聯合機率")
        with col3:
            st.metric("聯合全輸機率", f"{p_lose:.4f}" if p_lose is not None else "N/A",
                      help="所有市場同時為 No 的機率")

        # ── 自動解讀 ──
        st.subheader("📢 自動解讀")
        if edge > 0.05:
            st.success(f"🔥 **強烈正向 edge**！T-Copula 聯合機率比市場獨立看法高 **{multiplier:.1f} 倍**。"
                       f"市場低估這些事件「一起發生」的可能性 → **建議買 Yes**")
        elif edge > 0.02:
            st.info(f"✅ 有明顯正向 edge（高 {multiplier:.1f} 倍）。適合小注買 Yes，捕捉尾部相關性。")
        elif edge < -0.02:
            st.warning(f"⚠️ **負 edge**：Copula 機率比市場低，建議買 No 或觀望。")
        else:
            st.info("市場價格已接近 Copula 模擬，沒有明顯 edge，可觀望。")

        st.caption(f"模擬次數：{num_sim:,} 次 ｜ Copula：T（自由度 4）｜ 有效相關性強度：{effective_corr}")

        st.divider()

        # ==================== 進階分析區 ====================
        st.subheader("🔬 進階分析")
        st.caption("以下兩項分析提供額外參考視角，幫助你更全面理解市場定價。")

        # ── 分析一：Edge 信賴區間 ──
        with st.expander("📐 分析一：Edge 信賴區間（這個 edge 是真實訊號還是模擬雜訊？）"):
            st.markdown("""
**為什麼需要信賴區間？**

模擬本身是隨機抽樣，就像你丟 15 萬次硬幣，每次結果都會稍有不同。
信賴區間告訴你：這個 edge 數字的**可信範圍**是多少。

> 如果信賴區間包含 0（例如 -0.002 ~ +0.008），代表 edge 可能只是模擬誤差，**不建議依此下注**。
>
> 如果信賴區間完全在 0 以上（例如 +0.015 ~ +0.041），代表 edge 是穩定訊號，**可信度較高**。
            """)

            se_sweep = np.sqrt(p_sweep * (1 - p_sweep) / num_sim)
            se_indep = np.sqrt(p_indep * (1 - p_indep) / num_sim)
            se_edge  = np.sqrt(se_sweep**2 + se_indep**2)
            ci_lower = edge - 1.96 * se_edge
            ci_upper = edge + 1.96 * se_edge

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Edge 點估計", f"{edge:+.4f}")
            with c2:
                st.metric("95% CI 下界", f"{ci_lower:+.4f}")
            with c3:
                st.metric("95% CI 上界", f"{ci_upper:+.4f}")

            if ci_lower > 0:
                st.success("✅ 信賴區間完全在 0 以上，edge 為穩定正向訊號")
            elif ci_upper < 0:
                st.warning("⚠️ 信賴區間完全在 0 以下，edge 為穩定負向訊號")
            else:
                st.info("ℹ️ 信賴區間跨越 0，edge 可能是模擬誤差，建議增加模擬次數或謹慎觀望")

        # ── 分析二：YES+NO 一致性驗證 ──
        with st.expander("⚖️ 分析二：YES+NO 一致性驗證（市場定價是否健康？）"):
            st.markdown("""
**為什麼要驗證 YES+NO=1？**

Polymarket 的核心原理是：**YES + NO = 1**（就像一張一美元被撕成兩半）。

結算時，如果事件發生：YES = $1，NO = $0；如果沒發生：YES = $0，NO = $1。
所以不管結果如何，**兩張兌換券加起來永遠值 $1**。

正常市場的 YES 價 + NO 價應該等於 1。如果總和明顯偏離 1，可能代表：
- 市場流動性不足，買賣價差較大
- 數據來源抓到的是掛單價而非成交價

> ⚠️ 注意：在共享訂單簿機制下，**同一市場的 YES+NO < 1 套利是不存在的**。
> 當你掛出 YES 賣單時，系統同時在 NO 市場產生對應的買單，不平衡的訂單會被立即撮合。
> 偏差通常只反映買賣價差（bid-ask spread），而非真正的套利機會。
            """)

            parity_results = []
            for slug in slugs:
                try:
                    url = f"https://gamma-api.polymarket.com/markets?slug={slug}"
                    resp = requests.get(url, timeout=10)
                    data = resp.json()
                    if data:
                        m = data[0]
                        yes = m.get('yes_price') or m.get('last_trade_price')
                        no  = m.get('no_price')
                        if yes and no:
                            yes, no = float(yes), float(no)
                            total = yes + no
                            spread = abs(1 - total)
                            parity_results.append({
                                'slug': slug, 'yes': yes, 'no': no,
                                'total': total, 'spread': spread
                            })
                        elif yes:
                            parity_results.append({
                                'slug': slug, 'yes': float(yes), 'no': None,
                                'total': None, 'spread': None
                            })
                except Exception:
                    pass

            if parity_results:
                for r in parity_results:
                    cols = st.columns([3, 1, 1, 1, 2])
                    with cols[0]:
                        st.markdown(f"`{r['slug']}`")
                    with cols[1]:
                        st.markdown(f"YES: **{r['yes']:.3f}**")
                    with cols[2]:
                        st.markdown(f"NO: **{r['no']:.3f}**" if r['no'] is not None else "NO: —")
                    with cols[3]:
                        st.markdown(f"總和: **{r['total']:.3f}**" if r['total'] is not None else "總和: —")
                    with cols[4]:
                        if r['spread'] is not None:
                            if r['spread'] < 0.02:
                                st.success(f"✅ 正常（偏差 {r['spread']:.3f}）")
                            elif r['spread'] < 0.05:
                                st.warning(f"⚠️ 偏差略大（{r['spread']:.3f}）")
                            else:
                                st.error(f"❌ 偏差過大（{r['spread']:.3f}）")
                        else:
                            st.caption("缺少 NO 報價")
            else:
                st.info("無法取得 YES/NO 雙邊報價，可能該市場只提供單邊價格")

    else:
        st.error("❌ 解析失敗：無法從輸出中找到有效的機率數據")
        st.warning("請確認市場 Slug 是否正確，以及 API 是否有回傳數值")

    with st.expander("🔍 點擊查看原始 bot 輸出"):
        st.code(output)

# ==================== 側邊欄 ====================
with st.sidebar:
    st.header("📖 使用說明")
    st.markdown("""
1. 在上方填入 **2～5 個** Polymarket 市場 Slug
2. 設定相關性強度（或展開進階逐對設定）
3. 調整模擬次數（越高越精確）
4. 點擊「執行模擬」

**如何找到 Slug？**  
使用下方搜尋工具，或前往 Polymarket 市場頁面，URL 最後一段即為 Slug：  
`polymarket.com/event/`**`this-is-the-slug`**
    """)

    st.divider()

    # ==================== Slug 搜尋工具 ====================
    st.subheader("🔍 查詢有效 Slug（支援中文）")

    keyword = st.text_input("輸入關鍵字（中文或英文）", placeholder="e.g. 比特幣 / bitcoin / taiwan")

    if st.button("搜尋市場", use_container_width=True):
        if not keyword.strip():
            st.warning("請輸入關鍵字")
        else:
            search_keyword = keyword.strip()
            if re.search(r'[\u4e00-\u9fff]', search_keyword):
                try:
                    translated = GoogleTranslator(source='zh-TW', target='en').translate(search_keyword)
                    st.caption(f"🔄 翻譯為英文：`{translated}`")
                    search_keyword = translated
                except Exception as e:
                    st.warning(f"翻譯失敗，直接用原始關鍵字搜尋：{e}")

            try:
                url = f"https://gamma-api.polymarket.com/markets?q={search_keyword}&limit=10&active=true&closed=false"
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                markets = resp.json()

                if not markets:
                    st.warning("找不到相關活躍市場，請換個關鍵字")
                else:
                    st.success(f"找到 {len(markets)} 個市場：")
                    for m in markets:
                        slug = m.get('slug', '')
                        question = m.get('question', slug)
                        yes_price = m.get('yes_price') or m.get('last_trade_price')
                        price_str = f"　{float(yes_price):.1%}" if yes_price else "　（無報價）"
                        st.code(slug)
                        st.caption(f"{question}{price_str}")
            except Exception as e:
                st.error(f"搜尋失敗：{e}")

    st.divider()
    st.subheader("🔥 熱門 Slug 參考")
    st.info(
        "• `will-china-invade-taiwan-before-2027`\n"
        "• `china-x-taiwan-military-clash-before-2027`\n"
        "• `will-bitcoin-hit-100k-in-march`\n"
        "• `will-the-iranian-regime-fall-by-june-30`"
    )
    st.caption("Copula：T（自由度 4）｜ 相關性可於主畫面調整")