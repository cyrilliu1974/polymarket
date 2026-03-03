import streamlit as st
import subprocess
import re
import os
import sys
import requests
import numpy as np
from itertools import combinations
from deep_translator import GoogleTranslator
import json as _json
from datetime import datetime, timezone, timedelta

# ==================== 頁面配置(Page Config) / Page Config ====================
st.set_page_config(page_title="Polymarket Copula Bot (Bilingual)", layout="wide")
st.title("📊 Polymarket Copula Bot (聯合機率工具(Joint Probability Tool))")

# ==================== 兩個功能用 Tab 區隔(Two Features Separated by Tabs) / Tabs Layout ====================
tab1, tab2 = st.tabs([
    "🔗 Joint Probability Simulation)",
    "🏁 Tail Sweeper"
])

# ==============================================================
# TAB 1：Copula 聯合機率模擬(Joint Probability Simulation) / Copula Simulation
# ==============================================================
with tab1:
    st.markdown("**指定 2～5 個市場 Slug(Specify 2~5 market Slugs)，用 T-Copula 模擬聯合機率(Use T-Copula to simulate joint probability) / Simulate joint probabilities using T-Copula to find market edges.**")
    st.caption("適用場景(Applicable Scenarios)：跨事件套利(Cross-Event Arbitrage)、長線相關性分析(Long-Term Correlation Analysis) / Best for cross-event arbitrage & correlation analysis.")
    st.divider()

    st.subheader("🎯 市場 Slug 輸入(Market Slug Input) / Market Slug Input")
    st.caption("填入 Polymarket Slug(Fill in Polymarket Slug) (2-5 markets) / Enter 2-5 slugs from Polymarket.")

    default_slugs = [
        "will-trump-nominate-kevin-warsh-as-the-next-fed-chair",
        "russia-ukraine-ceasefire-before-gta-vi-554",
        "", "", ""
    ]
    slug_inputs = []
    for i in range(5):
        label_cn = f"Slug {i+1} {'（必填）(Required)' if i < 2 else '（選填）(Optional)'}"
        label_en = f"Slug {i+1} {'(Required)' if i < 2 else '(Optional)'}"
        val = st.text_input(
            f"{label_cn} / {label_en}",
            value=default_slugs[i],
            placeholder="e.g. will-bitcoin-hit-100k-in-march",
            key=f"slug_{i}"
        )
        slug_inputs.append(val.strip())

    st.subheader("⚙️ 模擬設定(Simulation Settings) / Simulation Settings")
    num_sim = st.slider(
        "模擬次數(Simulation Count) / Simulation Iterations",
        min_value=10000, max_value=500000, value=150000, step=10000,
        help="Higher means more precision but slower execution / 次數越高結果越精確(The higher the count the more accurate the result)，但執行時間越長(but execution time longer)"
    )
    st.caption(f"目前設定(Current Setting) / Current: {num_sim:,} 次 / iterations")

    st.subheader("🔗 相關性設定(Correlation Settings) / Correlation Settings")
    st.caption("相關性(Correlation)代表「兩個事件同時發生」(two events occur simultaneously)的傾向強度(tendency strength) / Correlation measures the tendency of events to occur together.")

    global_corr = st.slider(
        "全域相關性強度(Global Correlation Strength) / Global Correlation Strength",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="0.9 for highly related themes, 0.2 for low correlation / 同主題市場(Same theme markets)可設 0.9，不同主題(Different themes)可設 0.2"
    )

    active_slugs = [s for s in slug_inputs if s.strip()]
    pair_corr = {}
    if len(active_slugs) >= 2:
        with st.expander("⚙️ 進階：自訂每對市場相關性(Advanced: Customize Pairwise Market Correlation) / Advanced: Custom Pairwise Correlation"):
            st.caption("實際執行時將取所有配對之平均值(In actual execution will take average of all pairs) / Will use the mean of all pairs for the simulation.")
            for (i, j) in combinations(range(len(active_slugs)), 2):
                val = st.slider(
                    f"Pair {i+1} ↔ {j+1}: `{active_slugs[i]}` vs `{active_slugs[j]}`",
                    min_value=0.0, max_value=1.0, value=global_corr, step=0.05,
                    key=f"corr_{i}_{j}"
                )
                pair_corr[(i, j)] = val

    st.divider()
    run_btn = st.button("🚀 執行模擬 + 自動解析(Run Simulation + Auto Parse) / Run Simulation & Parse", type="primary", use_container_width=True, key="run_copula")

    if run_btn:
        slugs = [s for s in slug_inputs if s]
        invalid_slugs = [s for s in slugs if not re.match(r'^[\w-]+$', s)]
        if invalid_slugs:
            st.error(f"❌ Slug 格式錯誤(Slug Format Error) / Format Error: {invalid_slugs}")
            st.stop()
        if len(slugs) < 2:
            st.error("❌ 請至少填入 2 個有效的 Slug(Please fill in at least 2 valid Slugs) / Minimum 2 slugs required")
            st.stop()

        effective_corr = round(float(np.mean(list(pair_corr.values()))), 2) if pair_corr else global_corr
        markets_str = ",".join(slugs)
        args_list = [
            "--mode", "correlated_demo",
            "--markets", markets_str,
            "--copula", "t",
            "--num_sim", str(num_sim),
            "--corr_strength", str(effective_corr)
        ]
        st.info(f"📋 執行市場(Running Markets) / Markets: {' | '.join(slugs)} ｜ 有效相關性(Effective Correlation) / Effective Corr: {effective_corr}")

        current_env = os.environ.copy()
        current_env["PYTHONIOENCODING"] = "utf-8"
        with st.spinner(f"正在模擬(Running Simulation) {num_sim:,} 次 / Simulating, please wait..."):
            result = subprocess.run(
                [sys.executable, "polymarket_simulation_bot.py"] + args_list,
                capture_output=True, encoding="utf-8", errors="replace",
                env=current_env, timeout=600
            )

        output = result.stdout
        error_log = result.stderr

        if result.returncode != 0:
            st.error("❌ Bot 執行失敗(Bot Execution Failed) / Execution Failed")
            fail_matches = re.findall(r"❌ \[關鍵錯誤\](.*?)(?=❌|$)", output, re.DOTALL)
            if fail_matches:
                for msg in fail_matches:
                    st.warning(f"**診斷(Diagnosis) / Diagnostic:** {msg.strip()}")
            else:
                st.warning("請檢查完整日誌(Please check complete log) / Please check log below")
            with st.expander("🔍 查看完整日誌(View Complete Log) / View Full Log"):
                st.code(output + "\n" + error_log)
            st.stop()

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
            st.success("✅ 模擬完成(Simulation Completed) / Simulation Complete")

            if market_probs:
                st.subheader("📌 各市場現況(Each Market Current Status) / Market Status")
                prob_cols = st.columns(len(market_probs))
                for idx, (slug, prob_str) in enumerate(market_probs):
                    with prob_cols[idx]:
                        st.metric(label=slug, value=f"{prob_str}%")

            st.divider()
            st.subheader("📊 聯合機率模擬結果(Joint Probability Simulation Results) / Joint Probability Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("獨立相乘機率(Independent Multiplication Probability) / Independent Prob", f"{p_indep:.4f}", help="Probability if markets are independent / 假設各市場完全獨立時的聯合機率")
            with col2:
                st.metric("T-Copula 聯合掃全部(T-Copula Joint Sweep All) / Joint Prob (Sweep)", f"{p_sweep:.4f}",
                          delta=f"{edge:+.4f} ({edge*100:+.2f}%) x{multiplier:.3f}倍",
                          delta_color="normal" if edge >= 0 else "inverse",
                          help="Joint probability considering tail correlations / 考慮尾部相關性後的聯合機率")
            with col3:
                st.metric("聯合全輸機率(Joint All Lose Probability) / Joint Prob (All No)", f"{p_lose:.4f}" if p_lose is not None else "N/A",
                          help="Probability of all markets ending in 'No' / 所有市場同時為 No 的機率")

            st.subheader("📢 自動解讀(Automated Interpretation) / Automated Insight")
            if edge > 0.05:
                st.success(f"🔥 **強烈正向 Edge / Strong Positive Edge**! ({edge*100:+.2f}% x{multiplier:.3f}). Market underestimates joint occurrence. **Suggest: BUY YES** / 市場低估聯合發生可能 → 建議買 Yes")
            elif edge > 0.02:
                st.info(f"✅ 有明顯正向 Edge / Noticable Edge ({edge*100:+.2f}% x{multiplier:.3f}). Good for small YES bets to capture tail risk. / 適合小注買 Yes 捕捉尾部相關性")
            elif edge < -0.02:
                st.warning(f"⚠️ **負 Edge / Negative Edge**: Copula prob lower than market. Suggest BUY NO or Watch. / Copula 機率較低，建議買 No 或觀望")
            else:
                st.info("No significant edge. Market price aligns with simulation. / 市場定價接近模擬，建議觀望")
            st.caption(f"Simulations: {num_sim:,} | Copula: T (df=4) | Corr: {effective_corr}")

            # ── 相關性增益視覺化 ──
            st.markdown("**📊 相關性增益圖 / Correlation Gain Visualization**")
            st.caption("顯示獨立機率 vs T-Copula 聯合機率的差距 / Gap between independent vs correlated joint probability")
            _bar_max = min(max(p_sweep, p_indep) * 1.15, 1.0)
            col_viz1, col_viz2 = st.columns(2)
            with col_viz1:
                st.caption(f"獨立機率 / Independent: {p_indep:.4f}")
                st.progress(min(p_indep / _bar_max, 1.0))
            with col_viz2:
                st.caption(f"T-Copula 聯合機率 / Joint: {p_sweep:.4f}  {'▲' if edge >= 0 else '▼'} {abs(edge)*100:.2f}%")
                st.progress(min(p_sweep / _bar_max, 1.0))

            # ── Yes Bias 偵測 ──
            yes_bias_debug = []
            for _mp_slug, _prob_str in market_probs:
                try:
                    _yes_p = float(_prob_str) / 100
                    _resp = requests.get(f"https://gamma-api.polymarket.com/markets?slug={_mp_slug}", timeout=10)
                    _data = _resp.json()
                    if _data:
                        _end_raw = _data[0].get('endDate') or _data[0].get('end_date') or ''
                        if _end_raw:
                            if len(_end_raw) == 10: _end_raw += 'T00:00:00Z'
                            _end_dt = datetime.fromisoformat(_end_raw.replace('Z', '+00:00'))
                            _days_left = (_end_dt - datetime.now(timezone.utc)).days
                            yes_bias_debug.append((_mp_slug, _yes_p, _days_left))
                        else:
                            yes_bias_debug.append((_mp_slug, _yes_p, None))
                    else:
                        yes_bias_debug.append((_mp_slug, _yes_p, None))
                except Exception as _e:
                    yes_bias_debug.append((_mp_slug, None, f"Error: {_e}"))

            st.divider()
            st.subheader("🧠 Yes Bias 偵測(Yes Bias Detection) / Yes Bias Detection")
            with st.expander("什麼是 Yes Bias？(What is Yes Bias?) / What is Yes Bias?"):
                st.markdown("""
**預測市場普遍存在「Yes Bias」（正面結果高估偏差）**
- **情緒驅動**: 人們天生對正面結果更有期待感。
- **時間滯後**: 臨近結算日時 NO 的價值會自然增長，但價格反映較慢。
- **實戰啟示**: YES > 70% 且距結算 > 30 天時需注意。
                """)

            for _item in yes_bias_debug:
                _mp_slug, _yes_p, _days = _item
                if isinstance(_days, str):
                    st.caption(f"`{_mp_slug}` — {_days}")
                elif _days is None:
                    st.caption(f"`{_mp_slug}` — 無法取得日期 / No end date found")
                elif _yes_p is not None and _yes_p > 0.7 and _days > 30:
                    _avg_yes = sum(float(p)/100 for _, p in market_probs) / len(market_probs) if market_probs else 0.5
                    _annualized = (edge / _avg_yes) * (365 / _days) * 100 if _days > 0 and _avg_yes > 0 else 0
                    _bias_l1 = f"⚠️ **{_mp_slug}** | YES: {_yes_p:.0%} | Days left: {_days} | 預期年化 Edge: **{_annualized:.1f}%**"
                    _bias_l2 = f"市場可能存在 Yes Bias，建議考慮買 No。若選擇仍買 YES，建議搭配 Stop-Loss 指令或分批進場，防範 {_days} 天內突發性反轉。/ If buying YES anyway: use Stop-Loss or scale in gradually."
                    st.warning(_bias_l1 + chr(10)*2 + _bias_l2)
                elif _yes_p is not None and _yes_p > 0.7:
                    st.info(f"✅ `{_mp_slug}` — YES {_yes_p:.0%}，接近結算 ({_days}d)，影響較小。 / Near settlement, lower bias risk.")
                else:
                    st.info(f"✅ `{_mp_slug}` — YES {_yes_p:.0%}，未達 Bias 門檻。 / Below Yes Bias threshold.")

            # ── 綜合風險評級 ──
            _bias_slugs = {s for s, p, d in yes_bias_debug if isinstance(d, int) and p is not None and p > 0.7 and d > 30}
            st.divider()
            st.markdown("**🎯 綜合風險評級 / Composite Risk Rating**")
            if edge > 0.02 and _bias_slugs:
                _l1 = f"⚠️ **數學面看多，但籌碼面過熱 / Math bullish, but market overheated**"
                _l2 = f"T-Copula 顯示 {edge*100:+.2f}% 優勢 (x{multiplier:.3f})，但 {', '.join(_bias_slugs)} 已達高位，隱含回撤風險。"
                _l3 = "建議：對沖或減少注碼，勿全倉追多。/ Suggest: hedge or reduce position size."
                st.error(_l1 + chr(10)*2 + _l2 + chr(10) + _l3)
            elif edge > 0.02:
                st.success(f"✅ **正向 Edge 且無 Yes Bias 疑慮 / Positive edge, no Yes Bias** — 訊號較為乾淨，可考慮進場。/ Clean signal, consider entry.")
            elif edge < -0.02:
                st.warning(f"⚠️ **負 Edge / Negative Edge** — Copula 機率低於市場獨立估計，建議觀望或買 No。/ Watch or consider NO.")
            else:
                st.info("ℹ️ Edge 不顯著，市場定價合理，建議觀望。/ Edge not significant, market fairly priced.")

            st.divider()
            st.subheader("🔬 進階分析(Advanced Analysis) / Advanced Analysis")

            with st.expander("📐 分析一：Edge 信賴區間 / Analysis 1: Confidence Interval"):
                st.markdown("**為什麼需要信賴區間？/ Why CIs matter?**")
                st.caption("判斷 Edge 是真實訊號還是模擬雜訊 / Determine if Edge is a signal or simulation noise.")
                se_sweep = np.sqrt(p_sweep * (1 - p_sweep) / num_sim)
                se_indep = np.sqrt(p_indep * (1 - p_indep) / num_sim)
                se_edge  = np.sqrt(se_sweep**2 + se_indep**2)
                ci_lower = edge - 1.96 * se_edge
                ci_upper = edge + 1.96 * se_edge
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Edge 點估計 / Point Est", f"{edge:+.4f}")
                with c2: st.metric("95% CI 下界 / Lower", f"{ci_lower:+.4f}")
                with c3: st.metric("95% CI 上界 / Upper", f"{ci_upper:+.4f}")
                if ci_lower > 0:
                    st.success("✅ 穩定正向訊號 / Stable positive signal")
                    st.info(f"🛡️ 高可靠性訊號 / Signal Reliability: High — CI 下界 {ci_lower:+.4f} 完全在 0 以上，此 Edge 為真實正向訊號，非模擬隨機誤差。")
                elif ci_upper < 0:
                    st.warning("⚠️ 穩定負向訊號 / Stable negative signal")
                elif ci_lower > -0.005:
                    st.info(f"🟡 中可靠性訊號 / Signal Reliability: Medium — CI 跨越 0 但偏差不大 ({ci_lower:+.4f} ~ {ci_upper:+.4f})，可附加模擬次數確認。")
                else:
                    st.info("ℹ️ 低可靠性 / Signal Reliability: Low — CI 跨越 0 且偏差大，建議增加模擬次數或謹慎觀望。")

            with st.expander("⚖️ 分析二：YES+NO 一致性驗證 / Analysis 2: YES+NO Parity"):
                st.markdown("**為什麼要驗證 YES+NO=1？/ Why check parity?**")
                st.caption("正常市場 YES 價 + NO 價應接近 1。偏差通常反映買賣價差。 / Sum should be ~1. Deviations reflect bid-ask spreads.")
                import json as _pjson
                def _parse_yes_no(m):
                    try:
                        outcomes = _pjson.loads(m.get('outcomes') or '[]')
                        prices   = _pjson.loads(m.get('outcomePrices') or '[]')
                        if outcomes and prices and len(outcomes) == len(prices):
                            yes_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == 'yes'), None)
                            no_idx  = next((i for i, o in enumerate(outcomes) if str(o).lower() == 'no'),  None)
                            if yes_idx is not None and no_idx is not None:
                                return float(prices[yes_idx]), float(prices[no_idx])
                            if len(prices) == 2:
                                return float(prices[0]), float(prices[1])
                    except Exception:
                        pass
                    ltp = m.get('lastTradePrice') or m.get('last_trade_price')
                    return (float(ltp), None) if ltp else (None, None)

                parity_results = []
                for slug in slugs:
                    try:
                        resp = requests.get(f"https://gamma-api.polymarket.com/markets?slug={slug}", timeout=10)
                        data = resp.json()
                        if data:
                            yes, no = _parse_yes_no(data[0])
                            if yes is not None and no is not None:
                                total = yes + no
                                parity_results.append({'slug': slug, 'yes': yes, 'no': no, 'total': total, 'spread': abs(1 - total)})
                            elif yes is not None:
                                parity_results.append({'slug': slug, 'yes': yes, 'no': None, 'total': None, 'spread': None})
                    except Exception: pass
                if parity_results:
                    for r in parity_results:
                        cols = st.columns([3, 1, 1, 1, 2])
                        with cols[0]: st.markdown(f"`{r['slug']}`")
                        with cols[1]: st.markdown(f"YES: **{r['yes']:.3f}**")
                        with cols[2]: st.markdown(f"NO: **{r['no']:.3f}**" if r['no'] else "NO: —")
                        with cols[3]: st.markdown(f"Σ: **{r['total']:.3f}**" if r['total'] else "Σ: —")
                        with cols[4]:
                            if r['spread'] is not None:
                                if r['spread'] < 0.02: st.success(f"✅ 正常 / Normal ({r['spread']:.3f})")
                                elif r['spread'] < 0.05: st.warning(f"⚠️ 偏差略大 / High Spread ({r['spread']:.3f})")
                                else: st.error(f"❌ 偏差過大 / Gap Too Big ({r['spread']:.3f})")
                else: st.info("無法取得雙邊報價 / Could not fetch both prices")
        else:
            st.error("❌ 解析失敗(Parsing Failed) / Parsing Failed")

        with st.expander("🔍 查看原始 Bot 輸出(View Raw Bot Output) / View Raw Output"):
            st.code(output)

# ==============================================================
# TAB 2：掃尾盤篩選器(Tail Sweeper) / Tail Sweeper
# ==============================================================
with tab2:
    st.markdown("**自動掃描全市場(Auto scan entire market)，找出「即將結算 + YES 高價」/ Auto-scan all markets for 'near-settlement + high YES price' candidates.**")
    st.caption("獨立工具(Independent Tool) / Independent tool.")
    scan_direction = st.radio(
        "掃尾盤方向(Tail Sweep Direction) / Sweep Direction",
        options=["YES 尾盤(YES Tail)（事件即將發生(Event is about to happen)）", "NO 尾盤(NO Tail)（事件即將未發生(Event is about to not occur)）"],
        horizontal=True
    )
    is_yes_sweep = (scan_direction == "YES 尾盤(YES Tail)（事件即將發生(Event is about to happen)）")

    if is_yes_sweep:
        st.info("""
💡 **YES 尾盤**：當事件結果已幾乎確定，YES 價格停在 0.97 附近。買入等結算拿回 $1，賺取確定性收益。

⚠️ **風險**：黑天鵝事件（裁判反轉、比賽取消）可能讓 0.99 的籌碼瞬間歸零。建議單一市場最多投入總資金的 1/10。
        """)
    else:
        st.info("""
💡 **NO 尾盤**：當事件遲遲未發生，NO 價格會隨時間流逝自然升高，市場情緒卻滯後於現實。

策略：找 YES 價格極低（< 5%）且即將結算的市場，買入 NO，等待事件未發生得回 $1。

⚠️ **風險**：若事件突然發生，擔持的 NO 就會歸零。與 YES 尾盤的風險屬性相同。
        """)

    st.divider()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        hours_ahead = st.slider(
            "結算時間 (小時內) / Settlement within (hrs)",
            min_value=1, max_value=72, value=24, step=1,
            help="越小越近，確定性越高，黑天鵝時間窗口越短"
        )
    with col_s2:
        if is_yes_sweep:
            price_threshold = st.slider(
                "YES 價格下限 / Min YES Price",
                min_value=0.80, max_value=0.99, value=0.95, step=0.01,
                help="越高代表市場越篤定，但剩餘獲利空間也越小"
            )
        else:
            price_threshold = st.slider(
                "YES 價格上限（越低代表 NO 越篤定）/ Max YES Price",
                min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                help="設 0.05 代表只找 YES < 5% 的市場，也就是 NO > 95% 的市場"
            )
    scanner_limit = st.slider("顯示數量 / Results Limit", min_value=5, max_value=50, value=20, step=5)
    min_volume = st.slider(
        "最低 24h 成交量過濾 / Min Volume ($)",
        min_value=0, max_value=50000, value=500, step=500,
        help="過濾殭屍市場：成交量為 0 的市場通常掛單斷層，看得到吃不到。/ Filter zombie markets with no liquidity."
    )
    debug_scanner = st.checkbox("🐛 Debug 模式 / Debug Mode", value=False)

    st.divider()
    scan_btn = st.button("🔍 開始掃描(Begin Scanning) / Start Scanning", type="primary", use_container_width=True, key="run_scanner")

    if scan_btn:
        with st.spinner(f"掃描中 / Scanning future {hours_ahead}h..."):
            try:
                now = datetime.now(timezone.utc)
                deadline = now + timedelta(hours=hours_ahead)
                _raw_markets = []
                for _offset in range(0, 5000, 200):
                    _url = f"https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=200&offset={_offset}"
                    _r = requests.get(_url, timeout=15)
                    if not _r.ok: break
                    _batch = _r.json()
                    if not _batch: break
                    _raw_markets.extend(_batch)

                def parse_end_date(m):
                    for key in ('endDate', 'end_date', 'umaEndDate'):
                        raw = m.get(key) or ''
                        if not raw: continue
                        try:
                            if len(raw) == 10: raw += 'T00:00:00Z'
                            return datetime.fromisoformat(raw.replace('Z', '+00:00'))
                        except: continue
                    return None

                scan_markets = [m for m in _raw_markets if (d := parse_end_date(m)) and now <= d <= deadline]

                st.caption(f"🔍 抓取原始市場數：{len(_raw_markets)} 筆")
                st.caption(f"⏰ 現在時間（UTC）：{now.strftime('%Y-%m-%dT%H:%M:%SZ')}")
                st.caption(f"⏰ 截止時間（UTC）：{deadline.strftime('%Y-%m-%dT%H:%M:%SZ')}")
                st.caption(f"📋 時間過濾後剩餘：{len(scan_markets)} 筆")
                if _raw_markets:
                    for slug_s, ed in [(m.get('slug','?'), m.get('endDate','無')) for m in _raw_markets[:5]]:
                        st.caption(f"  └ {slug_s[:40]} → endDate: {ed}")

                def resolve_yes_price(m):
                    try:
                        outcomes = _json.loads(m.get('outcomes') or '[]')
                        prices   = _json.loads(m.get('outcomePrices') or '[]')
                        if outcomes and prices:
                            for i, o in enumerate(outcomes):
                                if str(o).lower() in ('yes', 'true', '1'): return float(prices[i])
                            return max(float(p) for p in prices)
                    except: pass
                    ltp = m.get('lastTradePrice') or m.get('last_trade_price')
                    return float(ltp) if ltp else None

                candidates = []
                for m in scan_markets:
                    yes_price = resolve_yes_price(m)
                    if yes_price is None:
                        continue
                    _vol_c = float(m.get('volume24hr') or m.get('volume') or 0)
                    if _vol_c < min_volume:
                        continue
                    if is_yes_sweep and yes_price >= price_threshold:
                        candidates.append({**m, '_yes_price_resolved': yes_price})
                    elif not is_yes_sweep and yes_price <= price_threshold:
                        candidates.append({**m, '_yes_price_resolved': yes_price})
                candidates = candidates[:scanner_limit]

                if not candidates:
                    if is_yes_sweep:
                        st.info(f"找不到符合條件的市場（未來 {hours_ahead}h 內結算且 YES ≥ {price_threshold:.0%}）")
                    else:
                        st.info(f"找不到符合條件的市場（未來 {hours_ahead}h 內結算且 YES ≤ {price_threshold:.0%}）")
                    st.caption("建議放寬條件後再試。體育賽事結束後是最容易找到的時機。")
                else:
                    if is_yes_sweep:
                        st.success(f"找到 {len(candidates)} 個 YES 尾盤候選")
                        st.caption("以下市場 YES 價格偏高且即將結算，進場前請自行判斷黑天鵝風險。")
                    else:
                        st.success(f"找到 {len(candidates)} 個 NO 尾盤候選")
                        st.caption("以下市場 YES 極低（事件幾乎不會發生），即將結算。買入 NO 等待事件未發生即可收回 $1。")
                    for m in candidates:
                        yes_price  = m.get('_yes_price_resolved', 0)
                        no_price   = 1 - yes_price
                        question   = m.get('question', m.get('slug', ''))
                        slug       = m.get('slug', '')
                        end_date   = m.get('endDateIso') or m.get('endDate') or ''
                        volume     = m.get('volume') or m.get('volume24hr') or 0
                        profit_pct = (1 - yes_price) * 100 if is_yes_sweep else yes_price * 100

                        _end_dt_c = parse_end_date(m)
                        _hours_left = max((_end_dt_c - now).total_seconds() / 3600, 0.1) if _end_dt_c else None
                        _daily_yield = (profit_pct / (_hours_left / 24)) if _hours_left else None

                        if is_yes_sweep:
                            _label = "✅ 確定性收尾盤" if yes_price >= 0.95 else ("⚠️ 高收益博弈" if yes_price >= 0.85 else "🔴 投機")
                        else:
                            _label = "✅ 確定性收尾盤" if yes_price <= 0.05 else ("⚠️ 高收益博弈" if yes_price <= 0.15 else "🔴 投機")

                        cols = st.columns([4, 1, 1, 1, 1])
                        with cols[0]:
                            st.markdown(f"**{question}**")
                            _end_str = str(end_date)[:16] if end_date else 'N/A'
                            _hrs_str = f"{_hours_left:.1f}h" if _hours_left else "?"
                            st.caption(f"`{slug}` ｜ 結算: {_end_str} (剩 {_hrs_str}) ｜ {_label}")
                        with cols[1]:
                            st.metric("YES 價格" if is_yes_sweep else "NO 價格", f"{yes_price:.3f}" if is_yes_sweep else f"{no_price:.3f}")
                        with cols[2]:
                            st.metric("獲利空間", f"{profit_pct:.1f}%")
                        with cols[3]:
                            st.metric("日均獲利潛力", f"{_daily_yield:.1f}%/d" if _daily_yield is not None else "N/A")
                        with cols[4]:
                            try: vol_str = f"${float(volume):,.0f}"
                            except: vol_str = "N/A"
                            st.metric("24h 成交量", vol_str)
                        st.divider()
            except Exception as e:
                st.error(f"掃描失敗 / Scan Failed: {e}")

# ==============================================================
# 側邊欄(Sidebar) / Sidebar
# ==============================================================
with st.sidebar:
    st.header("📖 Instructions / 使用說明")
    st.markdown("""
**1. Copula Simulation / 聯合機率模擬**
- Input 2-5 Slugs / 填入 2-5 個 Slug
- Set Correlation / 設定相關性
- Run / 執行獲取 Edge

**2. Tail Sweeper / 掃尾盤篩選器**
- Scan for near-settlement wins / 掃描即將結算之高勝率市場
- Check ROI & Volume / 確認收益與量能
                """)
    st.divider()
    st.subheader("🔍 Search Slugs / 查詢 Slug")
    keyword = st.text_input("關鍵字 (中/英) / Keyword", placeholder="e.g. bitcoin / 台灣")
    if st.button("搜尋市場 / Search", use_container_width=True):
        if keyword.strip():
            search_keyword = keyword.strip()

            # 偵測中文並自動翻譯
            if re.search(r'[\u4e00-\u9fff]', search_keyword):
                try:
                    translated = GoogleTranslator(source='zh-TW', target='en').translate(search_keyword)
                    st.caption(f"🔄 Translated: `{translated}`")
                    search_keyword = translated
                except:
                    pass

            try:
                # 去除虛詞，只保留有意義的關鍵詞
                STOPWORDS = {
                    'will','the','a','an','by','in','on','at','to','of','for',
                    'is','be','are','was','were','has','have','had','do','does',
                    'or','and','not','any','that','this','it','its','with','from',
                    'before','after','than','into','about','when','what','how',
                }
                all_terms = search_keyword.lower().split()
                # 有意義的詞（用於比對）
                key_terms = [t for t in all_terms if t not in STOPWORDS and len(t) > 1]
                # 若去完後剩 0 個詞，退回全部詞
                if not key_terms:
                    key_terms = all_terms

                def _match_term(term, text):
                    if len(term) <= 3:
                        return bool(re.search(r'(?<![a-z])' + re.escape(term) + r'(?![a-z])', text))
                    return term in text

                seen_slugs = set()
                valid_markets = []

                def _add(m, extra_text='', score=0):
                    slug = m.get('slug', '')
                    if slug and slug not in seen_slugs:
                        seen_slugs.add(slug)
                        valid_markets.append({
                            'slug': slug,
                            'display_name': m.get('question') or extra_text or slug,
                            'vol': float(m.get('volume24hr', 0) or 0),
                            'score': score,
                        })

                def _score(m, extra_text=''):
                    """回傳命中的關鍵詞數量（0 = 完全沒命中）"""
                    if not isinstance(m, dict):
                        return 0
                    slug = m.get('slug', '')
                    if not slug or slug in seen_slugs:
                        return 0
                    full_text = f"{m.get('question', '')} {slug} {extra_text}".lower()
                    return sum(_match_term(t, full_text) for t in key_terms)

                with st.spinner("🔍 搜尋中 / Searching..."):
                    # 軌道一：直接抓 polymarket.com 網站搜尋（與網站搜尋框同一引擎）
                    _web_slugs = []
                    try:
                        _q_encoded = requests.utils.quote(search_keyword)
                        _web_r = requests.get(
                            f"https://polymarket.com/search?_q={_q_encoded}",
                            timeout=15,
                            headers={"User-Agent": "Mozilla/5.0"}
                        )
                        if _web_r.ok:
                            # 從 HTML 中提取所有 /event/ 和 /markets/ 路徑下的 slug
                            _slugs_found = re.findall(
                                r'(?:polymarket\.com/(?:zh/)?(?:event|market)s?/)([\w-]+)',
                                _web_r.text
                            )
                            _web_slugs = list(dict.fromkeys(_slugs_found))  # 去重保序
                    except Exception:
                        pass

                    # 用抓到的 slug 去 gamma-api 補齊 question / volume
                    for _ws in _web_slugs:
                        try:
                            _r = requests.get(
                                f"https://gamma-api.polymarket.com/markets?slug={_ws}&limit=5",
                                timeout=10
                            )
                            if _r.ok:
                                for m in _r.json():
                                    s = _score(m)
                                    _add(m, score=max(s, 1))  # 網站搜尋命中視為至少 1 分
                        except Exception:
                            pass

                        # 若是 event slug，抓底下的子市場
                        try:
                            _r2 = requests.get(
                                f"https://gamma-api.polymarket.com/events?slug={_ws}&limit=1",
                                timeout=10
                            )
                            if _r2.ok:
                                for ev in _r2.json():
                                    ev_title = ev.get('title', '')
                                    for m in (ev.get('markets') or []):
                                        if isinstance(m, dict):
                                            if not m.get('question'):
                                                m['question'] = ev_title
                                            _add(m, extra_text=ev_title, score=2)
                        except Exception:
                            pass

                    # 軌道二：gamma-api ?q= 語意搜尋補充
                    for q in key_terms:
                        try:
                            r_m = requests.get(
                                f"https://gamma-api.polymarket.com/markets?q={requests.utils.quote(q)}&active=true&closed=false&limit=100",
                                timeout=10
                            )
                            if r_m.ok:
                                for m in r_m.json():
                                    s = _score(m)
                                    if s > 0: _add(m, score=s)

                            r_ev = requests.get(
                                f"https://gamma-api.polymarket.com/events?q={requests.utils.quote(q)}&active=true&closed=false&limit=50",
                                timeout=10
                            )
                            if r_ev.ok:
                                for ev in r_ev.json():
                                    ev_title = ev.get('title', '')
                                    for m in (ev.get('markets') or []):
                                        if isinstance(m, dict):
                                            if not m.get('question'):
                                                m['question'] = ev_title
                                            s = _score(m, extra_text=ev_title)
                                            if s > 0: _add(m, extra_text=ev_title, score=s)
                        except Exception:
                            pass

                # 先依命中關鍵詞數排序，同分再依成交量排
                valid_markets.sort(key=lambda x: (x['score'], x['vol']), reverse=True)
                top_results = valid_markets[:12]

                if not top_results:
                    st.warning(f"😶 找不到「{search_keyword}」的相關市場 / No results found")
                    st.markdown("""
**可能原因：**
- Polymarket 市場命名與搜尋詞不同（例如搜 `bubble`，實際 slug 是 `ai-industry-downturn`）
- 此市場為某 Event 的子選項，slug 與標題關鍵字不同
- 市場目前不在活躍列表中（已結算或流動性極低）

**建議策略：**

① **換同義詞或相關詞**
`bubble` → 試 `downturn` / `crash`；`agi` → 試 `artificial general` / `achieved`

② **去 Polymarket 官網直接找 slug**
前往 polymarket.com 搜尋，找到市場後複製網址最後一段即為 slug
例：`polymarket.com/event/ai-bubble-burst-by` → event slug 是 `ai-bubble-burst-by`

③ **直接輸入已知 slug**
將 slug 貼入上方 Copula 工具的輸入框直接使用，不需透過搜尋
""")
                else:
                    st.success(f"✅ 找到 {len(top_results)} 個市場 / Found {len(top_results)} markets:")
                    for res in top_results:
                        st.code(res['slug'])
                        st.caption(f"📝 {res['display_name']} | 💰 24h Vol: ${res['vol']:,.0f}")

            except Exception as e:
                st.error(f"連線異常 / Connection Error: {str(e)}")

    st.divider()
    st.subheader("🔥 Hot Slugs / 熱門參考")
    st.info("• `will-china-invade-taiwan-before-2027` \n • `will-bitcoin-hit-100k-in-march` ")