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

    # ── Slug 輸入(Slug Input) / Slug Input ──
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

    # ── 模擬設定(Simulation Settings) / Sim Settings ──
    st.subheader("⚙️ 模擬設定(Simulation Settings) / Simulation Settings")
    num_sim = st.slider(
        "模擬次數(Simulation Count) / Simulation Iterations",
        min_value=10000, max_value=500000, value=150000, step=10000,
        help="Higher means more precision but slower execution / 次數越高結果越精確(The higher the count the more accurate the result)，但執行時間越長(but execution time longer)"
    )
    st.caption(f"目前設定(Current Setting) / Current: {num_sim:,} 次 / iterations")

    # ── 相關性設定(Correlation Settings) / Correlation ──
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
                st.metric("獨立相乘機率(Independent Multiplication Probability) / Independent Prob", f"{p_indep:.4f}", help="Probability if markets are independent / 假設各市場完全獨立時的聯合機率(Assumed joint probability when each market is completely independent)")
            with col2:
                st.metric("T-Copula 聯合掃全部(T-Copula Joint Sweep All) / Joint Prob (Sweep)", f"{p_sweep:.4f}",
                          delta=f"{edge:+.4f} Edge",
                          delta_color="normal" if edge >= 0 else "inverse",
                          help="Joint probability considering tail correlations / 考慮尾部相關性後的聯合機率(Joint probability after considering tail correlations)")
            with col3:
                st.metric("聯合全輸機率(Joint All Lose Probability) / Joint Prob (All No)", f"{p_lose:.4f}" if p_lose is not None else "N/A",
                          help="Probability of all markets ending in 'No' / 所有市場同時為 No 的機率(Probability that all markets are No at the same time)")

            st.subheader("📢 自動解讀(Automated Interpretation) / Automated Insight")
            if edge > 0.05:
                st.success(f"🔥 **強烈正向 Edge / Strong Positive Edge**! (x{multiplier:.1f}). Market underestimates joint occurrence. **Suggest: BUY YES** / 市場低估聯合發生可能(Market underestimates joint occurrence possibility) → 建議買 Yes(Suggest: BUY YES)")
            elif edge > 0.02:
                st.info(f"✅ 有明顯正向 Edge / Noticable Edge (x{multiplier:.1f}). Good for small YES bets to capture tail risk. / 適合小注買 Yes 捕捉尾部相關性(Suitable for small bets on Yes to capture tail correlation)")
            elif edge < -0.02:
                st.warning(f"⚠️ **負 Edge / Negative Edge**: Copula prob lower than market. Suggest BUY NO or Watch. / Copula 機率較低(Copula probability lower)，建議買 No 或觀望(Suggest buy No or watch)")
            else:
                st.info("No significant edge. Market price aligns with simulation. / 市場定價接近模擬(Market pricing close to simulation)，建議觀望(Suggest watch)")
            st.caption(f"Simulations: {num_sim:,} | Copula: T (df=4) | Corr: {effective_corr}")

            # ── Yes Bias 偵測(Yes Bias Detection) / Yes Bias Detection ──
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
**預測市場普遍存在「Yes Bias」（正面結果高估偏差）(Prediction markets generally exist "Yes Bias" (positive result overestimation bias))**
- **情緒驅動(Sentiment Driven) / Sentiment Driven**: 人們天生對正面結果更有期待感(People naturally have more expectation for positive results)。
- **時間滯後(Time Lag) / Time Lag**: 臨近結算日時(When approaching settlement date) NO 的價值會自然增長(NO value will naturally grow)，但價格反映較慢(but price reflection is slower)。
- **實戰啟示(Action) / Action**: YES > 70% 且距結算 > 30 天時需注意(When YES > 70% and distance to settlement > 30 days need attention)。
                """)

            for _item in yes_bias_debug:
                _mp_slug, _yes_p, _days = _item
                if isinstance(_days, str):
                    st.caption(f"`{_mp_slug}` — {_days}")
                elif _days is None:
                    st.caption(f"`{_mp_slug}` — 無法取得日期(Cannot obtain date) / No end date found")
                elif _yes_p > 0.7 and _days > 30:
                    st.warning(f"⚠️ {_mp_slug} | YES: {_yes_p:.0%} | Days left: {_days} \n\n 市場可能存在 Yes Bias(Market may exist Yes Bias)，建議考慮買 No。(Suggest consider buy NO) / Possible Yes Bias, consider NO.")
                elif _yes_p > 0.7:
                    st.info(f"✅ `{_mp_slug}` — YES {_yes_p:.0%}，接近結算(Close to settlement) ({_days}d)，影響較小(Impact smaller)。 / Near settlement, lower bias risk.")
                else:
                    st.info(f"✅ `{_mp_slug}` — YES {_yes_p:.0%}，未達 Bias 門檻(Not reaching Bias threshold)。 / Below Yes Bias threshold.")

            st.divider()
            st.subheader("🔬 進階分析(Advanced Analysis) / Advanced Analysis")

            with st.expander("📐 分析一：Edge 信賴區間(Analysis 1: Edge Confidence Interval) / Analysis 1: Confidence Interval"):
                st.markdown("**為什麼需要信賴區間？(Why need confidence interval?) / Why CIs matter?**")
                st.caption("判斷 Edge 是真實訊號還是模擬雜訊(Determine if Edge is real signal or simulation noise) / Determine if Edge is a signal or simulation noise.")
                se_sweep = np.sqrt(p_sweep * (1 - p_sweep) / num_sim)
                se_indep = np.sqrt(p_indep * (1 - p_indep) / num_sim)
                se_edge  = np.sqrt(se_sweep**2 + se_indep**2)
                ci_lower = edge - 1.96 * se_edge
                ci_upper = edge + 1.96 * se_edge
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Edge 點估計(Edge Point Estimate) / Point Est", f"{edge:+.4f}")
                with c2: st.metric("95% CI 下界(95% CI Lower Bound) / Lower", f"{ci_lower:+.4f}")
                with c3: st.metric("95% CI 上界(95% CI Upper Bound) / Upper", f"{ci_upper:+.4f}")
                if ci_lower > 0: st.success("✅ 穩定正向訊號(Stable Positive Signal) / Stable positive signal")
                elif ci_upper < 0: st.warning("⚠️ 穩定負向訊號(Stable Negative Signal) / Stable negative signal")
                else: st.info("ℹ️ 可能是模擬誤差(May be simulation error)，建議增加次數(Suggest increase iterations) / Potentially noise, try more iterations")

            with st.expander("⚖️ 分析二：YES+NO 一致性驗證(Analysis 2: YES+NO Consistency Verification) / Analysis 2: YES+NO Parity"):
                st.markdown("**為什麼要驗證 YES+NO=1？(Why verify YES+NO=1?) / Why check parity?**")
                st.caption("正常市場(Normal market) YES 價 + NO 價應接近 1(YES price + NO price should approach 1)。偏差通常反映買賣價差(Deviation usually reflects buy-sell spread)。 / Sum should be ~1. Deviations reflect bid-ask spreads.")
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
                                if r['spread'] < 0.02: st.success(f"✅ 正常(Normal) / Normal ({r['spread']:.3f})")
                                elif r['spread'] < 0.05: st.warning(f"⚠️ 偏差略大(Slightly High Deviation) / High Spread ({r['spread']:.3f})")
                                else: st.error(f"❌ 偏差過大(Deviation Too Big) / Gap Too Big ({r['spread']:.3f})")
                else: st.info("無法取得雙邊報價(Cannot obtain both side quotes) / Could not fetch both prices")
        else:
            st.error("❌ 解析失敗(Parsing Failed) / Parsing Failed")

        with st.expander("🔍 查看原始 Bot 輸出(View Raw Bot Output) / View Raw Output"):
            st.code(output)

# ==============================================================
# TAB 2：掃尾盤篩選器(Tail Sweeper) / Tail Sweeper
# ==============================================================
with tab2:
    st.markdown("**自動掃描全市場(Auto scan entire market)，找出「即將結算 + YES 高價」(Find 'about to settle + high YES price')/ Auto-scan all markets for 'near-settlement + high YES price' candidates.**")
    st.caption("獨立工具(Independent Tool) / Independent tool.")
    scan_direction = st.radio(
        "掃尾盤方向(Tail Sweep Direction) / Sweep Direction",
        options=["YES 尾盤(YES Tail)（事件即將發生(Event is about to happen)）", "NO 尾盤(NO Tail)（事件即將未發生(Event is about to not occur)）"],
        horizontal=True
    )
    is_yes_sweep = (scan_direction == "YES 尾盤（事件即將發生）")

    if is_yes_sweep:
        st.info("""
💡 **YES 尾盤(YES Tail)**：當事件結果已幾乎確定(When event result is almost certain)，YES 價格停在 0.97 附近(YES price stays near 0.97)。買入等結算拿回 $1(Buy and wait for settlement to retrieve $1)，賺取確定性收益(Earn deterministic profit)。

⚠️ **風險(Risk)**：黑天鵝事件(Black Swan Event)（裁判反轉(Referee reversal)、比賽取消(Game cancellation)）可能讓 0.99 的籌碼瞬間歸零(May make 0.99 chips instantly zero)。建議單一市場最多投入總資金的 1/10(Suggest at most invest 1/10 of total capital per single market)。
        """)
    else:
        st.info("""
💡 **NO 尾盤(NO Tail)**：當事件遲遲未發生(When event has not occurred for a long time)，NO 價格會隨時間流逝自然升高(NO price will naturally rise as time passes)，市場情緒卻滯後於現實(Market sentiment lags behind reality)。

策略(Strategy)：找 YES 價格極低（< 5%）且即將結算的市場(Find markets with extremely low YES price (<5%) and about to settle)，買入 NO，等待事件未發生得回 $1(Buy NO, wait for event not to occur to get back $1)。

⚠️ **風險(Risk)**：若事件突然發生(If event suddenly occurs)，擔持的 NO 就會歸零(The held NO will go to zero)。與 YES 尾盤的風險屬性相同(Same risk characteristics as YES tail)。
        """)

    st.divider()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        hours_ahead = st.slider(
            "結算時間 (小時內)(Settlement Time (within hours)) / Settlement within (hrs)",
            min_value=1, max_value=72, value=24, step=1,
            help="越小越近(The smaller closer)，確定性越高(higher certainty)，黑天鵝時間窗口越短(black swan time window shorter)"
        )
    with col_s2:
        if is_yes_sweep:
            price_threshold = st.slider(
                "YES 價格下限(YES Price Lower Limit) / Min YES Price",
                min_value=0.80, max_value=0.99, value=0.95, step=0.01,
                help="越高代表市場越篤定(The higher means market more certain)，但剩餘獲利空間也越小(but remaining profit space also smaller)"
            )
        else:
            price_threshold = st.slider(
                "YES 價格上限（越低代表 NO 越篤定）(YES Price Upper Limit (lower means NO more certain)) / Max YES Price",
                min_value=0.01, max_value=0.20, value=0.05, step=0.01,
                help="設 0.05 代表只找 YES < 5% 的市場(Set 0.05 means only look for YES < 5% markets)，也就是 NO > 95% 的市場(that is NO > 95% markets)"
            )
    scanner_limit = st.slider("顯示數量(Display Quantity) / Results Limit", min_value=5, max_value=50, value=20, step=5)
    debug_scanner = st.checkbox("🐛 Debug 模式(Debug Mode) / Debug Mode", value=False)

    st.divider()
    scan_btn = st.button("🔍 開始掃描(Begin Scanning) / Start Scanning", type="primary", use_container_width=True, key="run_scanner")

    if scan_btn:
        with st.spinner(f"掃描中(Scanning) / Scanning future {hours_ahead}h..."):
            try:
                now = datetime.now(timezone.utc)
                deadline = now + timedelta(hours=hours_ahead)
                _raw_markets = []
                for _offset in range(0, 1000, 200):
                    _url = f"https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=200&offset={_offset}"
                    _r = requests.get(_url, timeout=15)
                    if not _r.ok: break
                    _batch = _r.json()
                    if not _batch: break
                    _raw_markets.extend(_batch)
                    if any((m.get('endDate') or '') > now.strftime('%Y-%m-%d') for m in _batch) and _offset >= 400: break

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
                    if is_yes_sweep and yes_price >= price_threshold:
                        candidates.append({**m, '_yes_price_resolved': yes_price})
                    elif not is_yes_sweep and yes_price <= price_threshold:
                        candidates.append({**m, '_yes_price_resolved': yes_price})
                candidates = candidates[:scanner_limit]

                if not candidates:
                    if is_yes_sweep:
                        st.info(f"找不到符合條件的市場(Cannot find qualifying markets)（未來 {hours_ahead}h 內結算且 YES ≥ {price_threshold:.0%}）")
                    else:
                        st.info(f"找不到符合條件的市場(Cannot find qualifying markets)（未來 {hours_ahead}h 內結算且 YES ≤ {price_threshold:.0%}）")
                    st.caption("建議放寬條件後再試(Suggest relax conditions and try again)。體育賽事結束後是最容易找到的時機。(After sports events end is the easiest timing to find)。")
                else:
                    if is_yes_sweep:
                        st.success(f"找到 {len(candidates)} 個 YES 尾盤候選(Found {len(candidates)} YES Tail Candidates)")
                        st.caption("以下市場(The following markets) YES 價格偏高(YES price biased high)且即將結算(and about to settle)，進場前請自行判斷黑天鵝風險。(Before entry please self-judge black swan risk)。")
                    else:
                        st.success(f"找到 {len(candidates)} 個 NO 尾盤候選(Found {len(candidates)} NO Tail Candidates)")
                        st.caption("以下市場(The following markets) YES 極低(YES extremely low)（事件幾乎不會發生(event almost will not occur)），即將結算(about to settle)。買入 NO 等待事件未發生即可收回 $1。(Buy NO wait for event not occur to recover $1)。")
                    for m in candidates:
                        yes_price = m.get('_yes_price_resolved', 0)
                        no_price  = 1 - yes_price
                        question  = m.get('question', m.get('slug', ''))
                        slug      = m.get('slug', '')
                        end_date  = m.get('endDateIso') or m.get('endDate') or '未知(Unknown)'
                        volume    = m.get('volume') or m.get('volume24hr') or 0
                        profit_pct = (1 - yes_price) * 100 if is_yes_sweep else yes_price * 100
                        cols = st.columns([4, 1, 1, 1])
                        with cols[0]:
                            st.markdown(f"**{question}**")
                            st.caption(f"`{slug}` ｜ 結算(Settlement): {str(end_date)[:16]}")
                        with cols[1]:
                            if is_yes_sweep:
                                st.metric("YES 價格(YES Price)", f"{yes_price:.3f}")
                            else:
                                st.metric("NO 價格(NO Price)", f"{no_price:.3f}")
                        with cols[2]:
                            st.metric("剩餘獲利空間(Remaining Profit Space)", f"{profit_pct:.1f}%")
                        with cols[3]:
                            try: vol_str = f"${float(volume):,.0f}"
                            except: vol_str = "N/A"
                            st.metric("交易量(Trading Volume)", vol_str)
                        st.divider()
            except Exception as e:
                st.error(f"掃描失敗(Scan Failed) / Scan Failed: {e}")

# ==============================================================
# 側邊欄(Sidebar) / Sidebar
# ==============================================================
with st.sidebar:
    st.header("📖 Instructions / 使用說明(Usage Instructions)")
    st.markdown("""
**1. Copula Simulation / 聯合機率模擬(Joint Probability Simulation)**
- Input 2-5 Slugs / 填入 2-5 個 Slug(Fill in 2-5 Slugs)
- Set Correlation / 設定相關性(Set Correlation)
- Run / 執行獲取 Edge(Run to obtain Edge)

**2. Tail Sweeper / 掃尾盤篩選器(Tail Sweeper)**
- Scan for near-settlement wins / 掃描即將結算之高勝率市場(Scan near-settlement high win-rate markets)
- Check ROI & Volume / 確認收益與量能(Confirm profit and volume)
                """)
    st.divider()
    st.subheader("🔍 Search Slugs / 查詢 Slug(Search Slug)")
    keyword = st.text_input("關鍵字 (中/英)(Keyword (Chinese/English)) / Keyword", placeholder="e.g. bitcoin / 台灣(Taiwan)")
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
                search_terms = search_keyword.lower().split()
                all_markets = []
                
                # 1. 戳 API 8 次，把全站最新的 1600 個活躍市場抓回記憶體
                with st.spinner("🚀 深度檢索全站活躍市場中 (約需 2-3 秒) / Deep searching active markets..."):
                    for offset in range(0, 1600, 200):
                        url = f"https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=200&offset={offset}"
                        resp = requests.get(url, timeout=10)
                        if not resp.ok: 
                            break
                        batch = resp.json()
                        if not batch or not isinstance(batch, list): 
                            break
                        all_markets.extend(batch)
                
                valid_markets = []
                seen_slugs = set()
                
                # 2. 用 Python 本地去比對字串
                for m in all_markets:
                    if not isinstance(m, dict): 
                        continue
                    slug = m.get('slug', '')
                    if not slug or slug in seen_slugs: 
                        continue
                        
                    m_question = m.get('question', '')
                    full_text = f"{m_question} {slug}".lower()
                    
                    if all(term in full_text for term in search_terms):
                        m_vol = float(m.get('volume24hr', 0) or 0)
                        valid_markets.append({
                            'slug': slug, 
                            'display_name': m_question or slug,
                            'vol': m_vol
                        })
                        seen_slugs.add(slug)

                # 依照交易量排序，取前 12 名
                valid_markets.sort(key=lambda x: x['vol'], reverse=True)
                top_results = valid_markets[:12]
                
                if not top_results:
                    st.info("沒有找到高度相關的市場，請嘗試更換或縮短關鍵字。 / No markets found.")
                else:
                    st.success(f"✅ 找到 {len(top_results)} 個市場 / Found {len(top_results)} markets:")
                    for res in top_results:
                        st.code(res['slug'])
                        st.caption(f"📝 {res['display_name']} | 💰 24h Vol: ${res['vol']:,.0f}")
                        
            except Exception as e:
                st.error(f"連線異常 / Connection Error: {str(e)}")

    st.divider()
    st.subheader("🔥 Hot Slugs / 熱門參考(Hot References)")
    st.info("• `will-china-invade-taiwan-before-2027` \n • `will-bitcoin-hit-100k-in-march` ")