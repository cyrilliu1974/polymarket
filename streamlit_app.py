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

# ==================== 兩個功能用 Tab 區隔 ====================
tab1, tab2 = st.tabs([
    "🔗 功能一：聯合機率模擬（Copula）",
    "🏁 功能二：掃尾盤篩選器（獨立工具）"
])

# ==============================================================
# TAB 1：Copula 聯合機率模擬
# ==============================================================
with tab1:
    st.markdown("**指定 2～5 個市場 Slug，用 T-Copula 模擬這些事件「同時發生」的聯合機率，找出市場定價錯誤的 edge。**")
    st.caption("適用場景：跨事件套利、長線相關性分析。與下方「掃尾盤」完全無關。")
    st.divider()

    # ── Slug 輸入 ──
    st.subheader("🎯 市場 Slug 輸入")
    st.caption("每個欄位填入一個 Polymarket 市場 Slug（必填 2 個，最多 5 個）")

    default_slugs = [
        "will-ken-paxton-win-the-2026-republican-primary",
        "russia-ukraine-ceasefire-before-gta-vi-554",
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

    # ── 模擬設定 ──
    st.subheader("⚙️ 模擬設定")
    num_sim = st.slider(
        "模擬次數",
        min_value=10000, max_value=500000, value=150000, step=10000,
        help="次數越高結果越精確，但執行時間越長"
    )
    st.caption(f"目前設定：{num_sim:,} 次")

    # ── 相關性設定 ──
    st.subheader("🔗 相關性設定")
    st.caption("相關性代表「兩個事件同時發生」的傾向強度。0 = 完全獨立，1 = 幾乎同進退")

    global_corr = st.slider(
        "全域相關性強度（套用到所有市場配對）",
        min_value=0.0, max_value=1.0, value=0.6, step=0.05,
        help="同主題市場可設 0.9，不同主題可設 0.2"
    )

    active_slugs = [s for s in slug_inputs if s.strip()]
    pair_corr = {}
    if len(active_slugs) >= 2:
        with st.expander("⚙️ 進階：自訂每對市場的相關性"):
            st.caption("展開後可針對每對市場個別設定，實際送出時取所有配對的平均值。")
            for (i, j) in combinations(range(len(active_slugs)), 2):
                val = st.slider(
                    f"Slug {i+1} ↔ Slug {j+1}：`{active_slugs[i]}` vs `{active_slugs[j]}`",
                    min_value=0.0, max_value=1.0, value=global_corr, step=0.05,
                    key=f"corr_{i}_{j}"
                )
                pair_corr[(i, j)] = val

    st.divider()
    run_btn = st.button("🚀 執行模擬 + 自動解析", type="primary", use_container_width=True, key="run_copula")

    if run_btn:
        slugs = [s for s in slug_inputs if s]
        invalid_slugs = [s for s in slugs if not re.match(r'^[\w-]+$', s)]
        if invalid_slugs:
            st.error(f"❌ 以下 Slug 格式錯誤：{invalid_slugs}")
            st.stop()
        if len(slugs) < 2:
            st.error("❌ 請至少填入 2 個有效的 Slug")
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

            if market_probs:
                st.subheader("📌 各市場現況")
                prob_cols = st.columns(len(market_probs))
                for idx, (slug, prob_str) in enumerate(market_probs):
                    with prob_cols[idx]:
                        st.metric(label=slug, value=f"{prob_str}%")

            st.divider()
            st.subheader("📊 聯合機率模擬結果")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("獨立相乘機率", f"{p_indep:.4f}", help="假設各市場完全獨立時的聯合機率")
            with col2:
                st.metric("T-Copula 聯合掃全部", f"{p_sweep:.4f}",
                          delta=f"{edge:+.4f} edge",
                          delta_color="normal" if edge >= 0 else "inverse",
                          help="考慮尾部相關性後的聯合機率")
            with col3:
                st.metric("聯合全輸機率", f"{p_lose:.4f}" if p_lose is not None else "N/A",
                          help="所有市場同時為 No 的機率")

            st.subheader("📢 自動解讀")
            if edge > 0.05:
                st.success(f"🔥 **強烈正向 edge**！T-Copula 聯合機率比市場獨立看法高 **{multiplier:.1f} 倍**。市場低估這些事件「一起發生」的可能性 → **建議買 Yes**")
            elif edge > 0.02:
                st.info(f"✅ 有明顯正向 edge（高 {multiplier:.1f} 倍）。適合小注買 Yes，捕捉尾部相關性。")
            elif edge < -0.02:
                st.warning(f"⚠️ **負 edge**：Copula 機率比市場低，建議買 No 或觀望。")
            else:
                st.info("市場價格已接近 Copula 模擬，沒有明顯 edge，可觀望。")
            st.caption(f"模擬次數：{num_sim:,} 次 ｜ Copula：T（自由度 4）｜ 有效相關性強度：{effective_corr}")

            # ── Yes Bias 偵測 ──
            from datetime import datetime, timezone
            yes_bias_flags = []
            yes_bias_debug = []  # 收集每個市場的檢查狀況
            for _mp_slug, _prob_str in market_probs:
                try:
                    _yes_p = float(_prob_str) / 100
                    _resp = requests.get(
                        f"https://gamma-api.polymarket.com/markets?slug={_mp_slug}",
                        timeout=10
                    )
                    _data = _resp.json()
                    if _data:
                        _end_raw = _data[0].get('endDate') or _data[0].get('end_date') or ''
                        if _end_raw:
                            if len(_end_raw) == 10:
                                _end_raw += 'T00:00:00Z'
                            _end_dt = datetime.fromisoformat(_end_raw.replace('Z', '+00:00'))
                            _days_left = (_end_dt - datetime.now(timezone.utc)).days
                            yes_bias_debug.append((_mp_slug, _yes_p, _days_left))
                            if _yes_p > 0.7 and _days_left > 30:
                                yes_bias_flags.append((_mp_slug, _yes_p, _days_left))
                        else:
                            yes_bias_debug.append((_mp_slug, _yes_p, None))
                    else:
                        yes_bias_debug.append((_mp_slug, _yes_p, None))
                except Exception as _e:
                    yes_bias_debug.append((_mp_slug, None, f"錯誤：{_e}"))

            st.divider()
            st.subheader("🧠 Yes Bias 偵測")
            with st.expander("什麼是 Yes Bias？點此展開說明"):
                st.markdown("""
**預測市場普遍存在「Yes Bias」（正面結果高估偏差）**

研究顯示，預測市場參與者傾向於高估正面事件（YES）發生的機率，原因包括：
- **情緒驅動**：人們天生對正面結果更有期待感，容易過度樂觀
- **時間滯後**：市場情緒往往慢於現實，臨近結算日時 NO 的價值會自然增長，但價格尚未反映
- **資訊不對稱**：散戶買 YES 的傾向遠高於買 NO，造成系統性高估

> **觸發條件**：YES 價格 > 70% 且距結算超過 30 天，才會出現警示。
> **實戰啟示**：符合條件的市場，考慮**買 NO** 或**等待更接近結算日**再入場，期望值可能更高。
                """)

            # 顯示每個市場的 Yes Bias 檢查結果
            for _item in yes_bias_debug:
                _mp_slug, _yes_p, _days = _item
                if isinstance(_days, str):
                    # 發生錯誤
                    st.caption(f"`{_mp_slug}` — {_days}")
                elif _days is None:
                    st.caption(f"`{_mp_slug}` — 無法取得結算日，跳過 Yes Bias 檢查")
                elif _yes_p > 0.7 and _days > 30:
                    _l1 = f"⚠️ {_mp_slug} | YES: {_yes_p:.0%} | 距結算還有 {_days} 天"
                    _l2 = "市場可能存在 Yes Bias，建議考慮買 No，或等結算日更近再入場"
                    st.warning(_l1 + chr(10) + chr(10) + _l2)
                elif _yes_p > 0.7:
                    st.info(f"✅ `{_mp_slug}` — YES {_yes_p:.0%}，距結算僅 **{_days} 天**，接近結算，Yes Bias 影響較小")
                else:
                    st.info(f"✅ `{_mp_slug}` — YES {_yes_p:.0%}，未達 Yes Bias 門檻（需 > 70%），無需特別注意")

            st.divider()
            st.subheader("🔬 進階分析")
            st.caption("以下兩項分析提供額外參考視角，幫助你更全面理解市場定價。")

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

            with st.expander("⚖️ 分析二：YES+NO 一致性驗證（市場定價是否健康？）"):
                st.markdown("""
**為什麼要驗證 YES+NO=1？**

Polymarket 的核心原理是：**YES + NO = 1**（就像一張一美元被撕成兩半）。

結算時，如果事件發生：YES = $1，NO = $0；如果沒發生：YES = $0，NO = $1。
所以不管結果如何，**兩張兌換券加起來永遠值 $1**。

正常市場的 YES 價 + NO 價應該等於 1。如果總和明顯偏離 1，可能代表市場流動性不足，買賣價差較大。

> ⚠️ 注意：在共享訂單簿機制下，**同一市場的 YES+NO < 1 套利是不存在的**。
> 偏差通常只反映買賣價差（bid-ask spread），而非真正的套利機會。
                """)
                import json as _pjson
                def _parse_yes_no(m):
                    """從 outcomePrices + outcomes 解析 YES/NO 價格"""
                    try:
                        outcomes = _pjson.loads(m.get('outcomes') or '[]')
                        prices   = _pjson.loads(m.get('outcomePrices') or '[]')
                        if outcomes and prices and len(outcomes) == len(prices):
                            yes_idx = next((i for i, o in enumerate(outcomes) if str(o).lower() == 'yes'), None)
                            no_idx  = next((i for i, o in enumerate(outcomes) if str(o).lower() == 'no'),  None)
                            if yes_idx is not None and no_idx is not None:
                                return float(prices[yes_idx]), float(prices[no_idx])
                            # 二元市場：第一個是 YES，第二個是 NO
                            if len(prices) == 2:
                                return float(prices[0]), float(prices[1])
                    except Exception:
                        pass
                    # fallback：lastTradePrice 只有 YES
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
                    except Exception:
                        pass
                if parity_results:
                    for r in parity_results:
                        cols = st.columns([3, 1, 1, 1, 2])
                        with cols[0]: st.markdown(f"`{r['slug']}`")
                        with cols[1]: st.markdown(f"YES: **{r['yes']:.3f}**")
                        with cols[2]: st.markdown(f"NO: **{r['no']:.3f}**" if r['no'] else "NO: —")
                        with cols[3]: st.markdown(f"總和: **{r['total']:.3f}**" if r['total'] else "總和: —")
                        with cols[4]:
                            if r['spread'] is not None:
                                if r['spread'] < 0.02: st.success(f"✅ 正常（偏差 {r['spread']:.3f}）")
                                elif r['spread'] < 0.05: st.warning(f"⚠️ 偏差略大（{r['spread']:.3f}）")
                                else: st.error(f"❌ 偏差過大（{r['spread']:.3f}）")
                            else: st.caption("缺少 NO 報價")
                else:
                    st.info("無法取得 YES/NO 雙邊報價，可能該市場只提供單邊價格")
        else:
            st.error("❌ 解析失敗：無法從輸出中找到有效的機率數據")
            st.warning("請確認市場 Slug 是否正確，以及 API 是否有回傳數值")

        with st.expander("🔍 點擊查看原始 bot 輸出"):
            st.code(output)

# ==============================================================
# TAB 2：掃尾盤篩選器（與 Copula 完全無關）
# ==============================================================
with tab2:
    st.markdown("**自動掃描 Polymarket 全市場，找出「即將結算 + YES 價格偏高」的套利候選。**")
    st.caption("此功能與左側 Slug 輸入完全無關，是獨立的全市場掃描工具。")
    st.info("""
💡 **什麼是掃尾盤？**

當一個事件結果已幾乎確定（比賽已結束、選舉已開票），但市場尚未正式結算時，
YES 價格會停留在接近 1 但不到 1 的區間（例如 0.97）。

此時買入，等待結算拿回 1 美元，賺取那最後幾個百分點的**確定性收益**。

⚠️ **風險提醒：** 黑天鵝事件（判定反轉、比賽取消）可能讓 0.99 的籌碼瞬間歸零。
資深玩家建議：**單一市場最多投入總資金的 1/10**，優先選擇幾小時內即將結算的市場。
    """)
    st.divider()

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        hours_ahead = st.slider(
            "篩選幾小時內結算的市場",
            min_value=1, max_value=72, value=24, step=1,
            help="設定越小，結算越近，確定性越高，黑天鵝時間窗口也越短"
        )
    with col_s2:
        min_yes_price = st.slider(
            "YES 價格下限",
            min_value=0.80, max_value=0.99, value=0.95, step=0.01,
            help="設定越高代表市場越篤定，但剩餘獲利空間也越小"
        )
    scanner_limit = st.slider("最多顯示幾個結果", min_value=5, max_value=50, value=20, step=5)
    debug_scanner = st.checkbox("🐛 Debug 模式（顯示 API 原始資料，用於排查問題）", value=False)

    st.divider()
    scan_btn = st.button("🔍 開始掃描尾盤市場", type="primary", use_container_width=True, key="run_scanner")

    if scan_btn:
        with st.spinner(f"掃描未來 {hours_ahead} 小時內結算、YES ≥ {min_yes_price:.0%} 的市場..."):
            try:
                import json as _json
                from datetime import datetime, timezone, timedelta
                now = datetime.now(timezone.utc)
                deadline = now + timedelta(hours=hours_ahead)

                _raw_markets = []
                for _offset in range(0, 1000, 200):
                    _url = (
                        f"https://gamma-api.polymarket.com/markets"
                        f"?active=true&closed=false&limit=200&offset={_offset}"
                    )
                    _r = requests.get(_url, timeout=15)
                    if not _r.ok:
                        break
                    _batch = _r.json()
                    if not _batch:
                        break
                    _raw_markets.extend(_batch)
                    _has_future = any((m.get('endDate') or '') > now.strftime('%Y-%m-%d') for m in _batch)
                    if _has_future and _offset >= 400:
                        break

                if debug_scanner:
                    st.warning("🐛 Debug 模式已開啟")
                    st.caption(f"API 回傳總筆數：{len(_raw_markets)}")
                    st.caption(f"目前時間（UTC）：`{now.strftime('%Y-%m-%dT%H:%M:%SZ')}`")
                    st.caption(f"篩選截止時間（UTC）：`{deadline.strftime('%Y-%m-%dT%H:%M:%SZ')}`")
                    if _raw_markets:
                        st.markdown("**前 5 筆的 endDate：**")
                        for _m in _raw_markets[:5]:
                            st.caption(f"`{_m.get('slug','')}` → endDate: `{_m.get('endDate','無')}` | outcomePrices: `{_m.get('outcomePrices','無')}`")
                        st.markdown("**第一筆價格欄位：**")
                        st.json({k: v for k, v in _raw_markets[0].items() if any(x in k.lower() for x in ['price','outcome','trade','last'])})

                def parse_end_date(m):
                    for key in ('endDate', 'end_date', 'umaEndDate'):
                        raw = m.get(key) or ''
                        if not raw:
                            continue
                        try:
                            if len(raw) == 10:
                                raw += 'T00:00:00Z'
                            return datetime.fromisoformat(raw.replace('Z', '+00:00'))
                        except Exception:
                            continue
                    return None

                scan_markets = [m for m in _raw_markets if (d := parse_end_date(m)) and now <= d <= deadline]

                if debug_scanner:
                    st.caption(f"時間過濾後剩餘：{len(scan_markets)} 筆")

                def resolve_yes_price(m):
                    try:
                        outcomes = _json.loads(m.get('outcomes') or '[]')
                        prices   = _json.loads(m.get('outcomePrices') or '[]')
                        if outcomes and prices:
                            for i, o in enumerate(outcomes):
                                if str(o).lower() in ('yes', 'true', '1'):
                                    return float(prices[i])
                            return max(float(p) for p in prices)
                    except Exception:
                        pass
                    ltp = m.get('lastTradePrice') or m.get('last_trade_price')
                    return float(ltp) if ltp else None

                candidates = []
                for m in scan_markets:
                    yes_price = resolve_yes_price(m)
                    if yes_price is not None and yes_price >= min_yes_price:
                        candidates.append({**m, '_yes_price_resolved': yes_price})
                candidates = candidates[:scanner_limit]

                if not candidates:
                    st.info(f"目前找不到符合條件的市場（未來 {hours_ahead}h 內結算且 YES ≥ {min_yes_price:.0%}）")
                    st.caption("建議：放寬條件（延長時間至 72h、降低 YES 門檻至 0.80）後再試。體育賽事結束後是最容易找到的時機。")
                else:
                    st.success(f"找到 {len(candidates)} 個候選市場")
                    st.caption("以下市場 YES 價格偏高且即將結算，是否值得進場，仍需自行判斷黑天鵝風險。")
                    for m in candidates:
                        yes_price = m.get('_yes_price_resolved', 0)
                        question  = m.get('question', m.get('slug', ''))
                        slug      = m.get('slug', '')
                        end_date  = m.get('endDateIso') or m.get('endDate') or '未知'
                        volume    = m.get('volume') or m.get('volume24hr') or 0
                        profit_pct = (1 - yes_price) * 100
                        cols = st.columns([4, 1, 1, 1])
                        with cols[0]:
                            st.markdown(f"**{question}**")
                            st.caption(f"`{slug}`　結算：{str(end_date)[:16]}")
                        with cols[1]:
                            st.metric("YES 價格", f"{yes_price:.3f}")
                        with cols[2]:
                            st.metric("剩餘獲利空間", f"{profit_pct:.1f}%")
                        with cols[3]:
                            try:
                                vol_str = f"${float(volume):,.0f}"
                            except:
                                vol_str = "N/A"
                            st.metric("交易量", vol_str)
                        st.divider()

            except Exception as e:
                st.error(f"掃尾盤失敗：{e}")

# ==============================================================
# 側邊欄
# ==============================================================
with st.sidebar:
    st.header("📖 使用說明")
    st.markdown("""
**功能一：聯合機率模擬**
1. 填入 2～5 個市場 Slug
2. 設定相關性強度
3. 調整模擬次數
4. 點擊「執行模擬」

**功能二：掃尾盤篩選器**
1. 切換到右側頁籤
2. 設定時間與價格門檻
3. 點擊「開始掃描」

兩個功能完全獨立，互不影響。
    """)
    st.divider()
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
                    st.warning(f"翻譯失敗：{e}")
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