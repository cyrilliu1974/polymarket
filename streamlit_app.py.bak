import streamlit as st
import subprocess
import re
import os
import requests
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
    required_mark = "\\*" if i < 2 else "（選填）"
    label = f"Slug {i+1} {'（必填）' if i < 2 else '（選填）'}"
    val = st.text_input(
        label,
        value=default_slugs[i],
        placeholder=f"e.g. will-bitcoin-hit-100k-in-march",
        key=f"slug_{i}"
    )
    slug_inputs.append(val.strip())

# ==================== 模擬次數 ====================
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

# ==================== 執行按鈕 ====================
st.divider()
run_btn = st.button("🚀 執行模擬 + 自動解析", type="primary", use_container_width=True)

if run_btn:
    # 過濾空白欄位
    slugs = [s for s in slug_inputs if s]

    # 驗證 slug 格式（只允許英文、數字、連字號）
    invalid_slugs = [s for s in slugs if not re.match(r'^[\w-]+$', s)]
    if invalid_slugs:
        st.error(f"❌ 以下 Slug 格式錯誤（只允許英文、數字、連字號）：{invalid_slugs}")
        st.stop()

    if len(slugs) < 2:
        st.error("❌ 請至少填入 2 個有效的 Slug")
        st.stop()

    markets_str = ",".join(slugs)

    # 固定參數：mode=correlated_demo, copula=t, corr_strength=0.6
    args_list = [
        "--mode", "correlated_demo",
        "--markets", markets_str,
        "--copula", "t",
        "--num_sim", str(num_sim),
        "--corr_strength", "0.6"
    ]

    st.info(f"📋 執行市場：{' | '.join(slugs)}")

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

        # 嘗試找出具體哪個 slug 有問題
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

    # 抓各市場個別機率
    market_probs = re.findall(r"✅ 抓到 ([\w-]+)：機率 ([\d.]+%)", output)

    if p_indep is not None and p_sweep is not None:
        edge = p_sweep - p_indep
        multiplier = p_sweep / p_indep if p_indep > 0 else float('inf')

        st.success("✅ 模擬完成，以下是解析結果")

        # 各市場個別機率
        if market_probs:
            st.subheader("📌 各市場現況")
            prob_cols = st.columns(len(market_probs))
            for idx, (slug, prob_str) in enumerate(market_probs):
                with prob_cols[idx]:
                    st.metric(label=slug, value=prob_str)

        st.divider()

        # 聯合機率結果
        st.subheader("📊 聯合機率模擬結果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("獨立相乘機率", f"{p_indep:.4f}",
                      help="假設各市場完全獨立時的聯合機率")
        with col2:
            delta_color = "normal" if edge >= 0 else "inverse"
            st.metric("T-Copula 聯合掃全部", f"{p_sweep:.4f}",
                      delta=f"{edge:+.4f} edge", delta_color=delta_color,
                      help="考慮尾部相關性後的聯合機率")
        with col3:
            st.metric("聯合全輸機率", f"{p_lose:.4f}" if p_lose is not None else "N/A",
                      help="所有市場同時為 No 的機率")

        # 自動解讀
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

        st.caption(f"模擬次數：{num_sim:,} 次 ｜ Copula：T（自由度 4）｜ 相關性強度：0.6")

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
    2. 調整模擬次數（越高越精確）
    3. 點擊「執行模擬」

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

            # 偵測是否含中文，若是則先翻譯
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
    st.caption("固定參數：Copula = T（自由度 4）｜ 相關性 = 0.6")