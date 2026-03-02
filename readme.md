這份專案結合了金融工程與預測市場分析，核心在於利用 Copula 模型捕捉 Polymarket 上多個事件之間的「尾部相關性」（Tail Dependence）。以下是為此專案撰寫的 `README.md` 檔案，包含中英文對照與技術分析。



---



\# 📊 Polymarket Copula \& Tail Sweeper Bot



這是一個基於 Streamlit 的全方位預測市場分析工具，旨在協助使用者識別 Polymarket 上的市場利差（Edge）並自動篩選即將結算的獲利機會。

This is a comprehensive Streamlit-based prediction market analysis tool designed to help users identify market edges on Polymarket and automatically screen for near-settlement profit opportunities.



\## 核心功能 | Key Features



\### 1. 聯合機率模擬 (T-Copula) | Joint Probability Simulation



\* \*\*多事件分析\*\*：支援 2 至 5 個市場 Slug 的同步分析。

\* \*\*Multi-Event Analysis\*\*: Supports simultaneous analysis of 2 to 5 market slugs.





\* \*\*尾部相關性捕捉\*\*：使用 T-Copula 算法模擬事件間的非線性相關性，這比單純的機率相乘更符合現實世界的極端連動情況。

\* \*\*Tail Dependence Capture\*\*: Utilizes T-Copula algorithms to simulate non-linear correlations between events, which more accurately reflects real-world extreme correlations than simple probability multiplication.





\* \*\*自動利差識別\*\*：比較「市場獨立機率」與「Copula 聯合機率」，找出被市場低估的連動機會。

\* \*\*Automated Edge Identification\*\*: Compares "Market Independent Probability" with "Copula Joint Probability" to find undervalued correlation opportunities.







\### 2. 掃尾盤篩選器 | Tail Sweeper



\* \*\*YES/NO 尾盤掃描\*\*：自動檢索全站即將在 1 至 72 小時內結算的市場。

\* \*\*YES/NO Tail Scanning\*\*: Automatically retrieves markets site-wide set to settle within 1 to 72 hours.





\* \*\*確定性收益\*\*：針對價格 >0.95 (YES) 或 <0.05 (YES) 的市場進行篩選，捕捉最後的結算利潤。

\* \*\*Deterministic Yield\*\*: Screens for markets with prices >0.95 (YES) or <0.05 (YES) to capture final settlement profits.







\### 3. 高級模擬引擎 | Advanced Simulation Engine



\* \*\*粒子濾波器 (Particle Filter)\*\*：動態追蹤事件（如選舉之夜）的即時機率走勢。

\* \*\*Particle Filter\*\*: Dynamically tracks real-time probability trends for events like election nights.





\* \*\*罕見事件重要性採樣 (Importance Sampling)\*\*：優化低機率事件（黑天鵝）的模擬效率。

\* \*\*Importance Sampling\*\*: Optimizes simulation efficiency for low-probability (Black Swan) events.







---



\## 技術架構 | Technical Architecture



此工具由兩個核心模組組成：

This tool consists of two core modules:



1\. \*\*`polymarket\_simulation\_bot.py`\*\*: 後端引擎，負責 Monte Carlo 模擬、Copula 計算及 Polymarket API 數據抓取。

The backend engine responsible for Monte Carlo simulations, Copula calculations, and Polymarket API data fetching.

2\. \*\*`streamlit\_app.py`\*\*: 前端界面，提供直觀的參數調整、自動翻譯關鍵字及視覺化結果。

The frontend interface providing intuitive parameter adjustments, automated keyword translation, and visualized results.



---



\## 安裝與執行 | Installation \& Execution



\### 環境需求 | Prerequisites



\* Python 3.11+

\* 依賴套件見 `requirements.txt`：`streamlit`, `numpy`, `scipy`, `requests`, `deep-translator`。



\### 啟動指令 | Start Command



```bash

pip install -r requirements.txt

streamlit run streamlit\_app.py



```



---



\## 坦率現實主義分析：利弊評估 | Frank Realism: Pros \& Cons



基於冷酷的事實分析，使用此工具需考慮以下平衡觀點：

Based on cold factual analysis, users must consider the following balanced perspective:



| 優點 (Pros) | 缺點與風險 (Cons \& Risks) |

| --- | --- |

| \*\*科學定價\*\*：超越直覺，利用 Copula 模型量化跨事件的風險連動。 | \*\*模型假設誤差\*\*：相關性參數（Corr Strength）為手動設定，若設定錯誤會導致誤導性結論。 |

| \*\*效率提升\*\*：自動化掃描即將結算的市場，節省大量人工搜尋時間。 | \*\*黑天鵝風險\*\*：在「掃尾盤」時，極端突發事件可能導致 0.99 的價格瞬間歸零。 |

| \*\*Yes Bias 偵測\*\*：自動提醒使用者避開情緒驅動的高估市場。 | \*\*API 延遲\*\*：數據抓取可能存在數秒延遲，在高波動市場可能錯失時機。 |



---



\## 批判性思考與限制 | Critical Thinking \& Limitations



1\. \*\*相關性不等於因果關係\*\*：本工具模擬的是統計上的相關性。如果兩個事件僅在統計上相關但缺乏邏輯上的因果連動，其模擬出的「Edge」可能只是幻覺。

\*\*Correlation $\\neq$ Causation\*\*: This tool simulates statistical correlations. If two events are statistically correlated but lack logical causal links, the simulated "Edge" might be an illusion.

2\. \*\*市場深度（Liquidity）\*\*：模擬結果可能顯示巨大的利差，但若該市場交易量過低，使用者可能無法在不大幅影響價格的情況下完成建倉。

\*\*Market Depth\*\*: Simulations may show significant edges, but if market liquidity is low, users may not be able to enter positions without significantly impacting the price.

3\. \*\*Yes Bias 侷限性\*\*：雖然工具能偵測 Yes Bias，但特定事件（如法律判決）的機率分佈並非對稱，單純的偏誤偵測無法取代深入的研究。

\*\*Yes Bias Limitations\*\*: While the tool detects Yes Bias, the probability distribution of specific events (e.g., legal rulings) is often asymmetric; simple bias detection cannot replace in-depth research.

