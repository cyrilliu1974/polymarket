import argparse
import requests
import numpy as np
import os
import re
import argparse
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
from scipy.stats import norm, t as t_dist
from scipy.special import expit, logit
from collections import deque

# ==================== 原始 function 保持不變 ====================

def simulate_binary_contract(S0, K, mu, sigma, T, N_paths=100_000):
    Z = np.random.standard_normal(N_paths)
    S_T = S0 * np.exp((mu - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = (S_T > K).astype(float)
    p_hat = payoffs.mean()
    se = np.sqrt(p_hat * (1 - p_hat) / N_paths)
    ci_lower = p_hat - 1.96 * se
    ci_upper = p_hat + 1.96 * se
    return {'probability': p_hat, 'std_error': se, 'ci_95': (ci_lower, ci_upper), 'N_paths': N_paths}

def brier_score(predictions, outcomes):
    return np.mean((np.array(predictions) - np.array(outcomes))**2)

def rare_event_IS(S0, K_crash, sigma, T, N_paths=100_000):
    K = S0 * (1 - K_crash)
    mu_original = -0.5 * sigma**2
    log_threshold = np.log(K / S0)
    mu_tilt = log_threshold / T
    Z = np.random.standard_normal(N_paths)
    log_returns_tilted = mu_tilt * T + sigma * np.sqrt(T) * Z
    S_T_tilted = S0 * np.exp(log_returns_tilted)
    log_returns_original = mu_original * T + sigma * np.sqrt(T) * Z
    log_LR = (
        -0.5 * ((log_returns_tilted - mu_original * T) / (sigma * np.sqrt(T)))**2
        + 0.5 * ((log_returns_tilted - mu_tilt * T) / (sigma * np.sqrt(T)))**2
    )
    LR = np.exp(log_LR)
    payoffs = (S_T_tilted < K).astype(float)
    is_estimates = payoffs * LR
    p_IS = is_estimates.mean()
    se_IS = is_estimates.std() / np.sqrt(N_paths)
    Z_crude = np.random.standard_normal(N_paths)
    S_T_crude = S0 * np.exp(mu_original * T + sigma * np.sqrt(T) * Z_crude)
    p_crude = (S_T_crude < K).mean()
    se_crude = np.sqrt(p_crude * (1 - p_crude) / N_paths) if p_crude > 0 else float('inf')
    return {
        'p_IS': p_IS, 'se_IS': se_IS,
        'p_crude': p_crude, 'se_crude': se_crude,
        'variance_reduction': (se_crude / se_IS)**2 if se_IS > 0 else float('inf')
    }

class PredictionMarketParticleFilter:
    def __init__(self, N_particles=5000, prior_prob=0.5, process_vol=0.05, obs_noise=0.03):
        self.N = N_particles
        self.process_vol = process_vol
        self.obs_noise = obs_noise
        logit_prior = logit(prior_prob)
        self.logit_particles = logit_prior + np.random.normal(0, 0.5, N_particles)
        self.weights = np.ones(N_particles) / N_particles
        self.history = []
    
    def update(self, observed_price):
        noise = np.random.normal(0, self.process_vol, self.N)
        self.logit_particles += noise
        prob_particles = expit(self.logit_particles)
        log_likelihood = -0.5 * ((observed_price - prob_particles) / self.obs_noise)**2
        log_weights = np.log(self.weights + 1e-300) + log_likelihood
        log_weights -= log_weights.max()
        self.weights = np.exp(log_weights)
        self.weights /= self.weights.sum()
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.N / 2:
            self._systematic_resample()
        self.history.append(self.estimate())
    
    def _systematic_resample(self):
        cumsum = np.cumsum(self.weights)
        u = (np.arange(self.N) + np.random.uniform()) / self.N
        indices = np.searchsorted(cumsum, u)
        self.logit_particles = self.logit_particles[indices]
        self.weights = np.ones(self.N) / self.N
    
    def estimate(self):
        probs = expit(self.logit_particles)
        return np.average(probs, weights=self.weights)
    
    def credible_interval(self, alpha=0.05):
        probs = expit(self.logit_particles)
        sorted_idx = np.argsort(probs)
        sorted_probs = probs[sorted_idx]
        sorted_weights = self.weights[sorted_idx]
        cumw = np.cumsum(sorted_weights)
        lower = sorted_probs[np.searchsorted(cumw, alpha/2)]
        upper = sorted_probs[np.searchsorted(cumw, 1 - alpha/2)]
        return lower, upper

def stratified_binary_mc(S0, K, sigma, T, J=10, N_total=100_000):
    n_per_stratum = N_total // J
    estimates = []
    for j in range(J):
        U = np.random.uniform(j/J, (j+1)/J, n_per_stratum)
        Z = norm.ppf(U)
        S_T = S0 * np.exp((-0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        stratum_mean = (S_T > K).mean()
        estimates.append(stratum_mean)
    p_stratified = np.mean(estimates)
    se_stratified = np.std(estimates) / np.sqrt(J)
    return p_stratified, se_stratified

def simulate_correlated_outcomes_gaussian(probs, corr_matrix, N=100_000):
    d = len(probs)
    L = np.linalg.cholesky(corr_matrix)
    Z = np.random.standard_normal((N, d))
    X = Z @ L.T
    U = norm.cdf(X)
    outcomes = (U < np.array(probs)).astype(int)
    return outcomes

def simulate_correlated_outcomes_t(probs, corr_matrix, nu=4, N=100_000):
    d = len(probs)
    L = np.linalg.cholesky(corr_matrix)
    Z = np.random.standard_normal((N, d))
    X = Z @ L.T
    S = np.random.chisquare(nu, N) / nu
    T = X / np.sqrt(S[:, None])
    U = t_dist.cdf(T, nu)
    outcomes = (U < np.array(probs)).astype(int)
    return outcomes

def simulate_correlated_outcomes_clayton(probs, theta=2.0, N=100_000):
    V = np.random.gamma(1/theta, 1, N)
    E = np.random.exponential(1, (N, len(probs)))
    U = (1 + E / V[:, None])**(-1/theta)
    outcomes = (U < np.array(probs)).astype(int)
    return outcomes

class PredictionMarketABM:
    def __init__(self, true_prob, n_informed=10, n_noise=50, n_mm=5):
        self.true_prob = true_prob
        self.price = 0.50
        self.price_history = [self.price]
        self.best_bid = 0.49
        self.best_ask = 0.51
        self.n_informed = n_informed
        self.n_noise = n_noise
        self.n_mm = n_mm
        self.volume = 0
        self.informed_pnl = 0
        self.noise_pnl = 0
    
    def step(self):
        total = self.n_informed + self.n_noise + self.n_mm
        r = np.random.random()
        if r < self.n_informed / total:
            self._informed_trade()
        elif r < (self.n_informed + self.n_noise) / total:
            self._noise_trade()
        else:
            self._mm_update()
        self.price_history.append(self.price)
    
    def _informed_trade(self):
        signal = self.true_prob + np.random.normal(0, 0.02)
        if signal > self.best_ask + 0.01:
            size = min(0.1, abs(signal - self.price) * 2)
            self.price += size * self._kyle_lambda()
            self.volume += size
            self.informed_pnl += (self.true_prob - self.best_ask) * size
        elif signal < self.best_bid - 0.01:
            size = min(0.1, abs(self.price - signal) * 2)
            self.price -= size * self._kyle_lambda()
            self.volume += size
            self.informed_pnl += (self.best_bid - self.true_prob) * size
        self.price = np.clip(self.price, 0.01, 0.99)
        self._update_book()
    
    def _noise_trade(self):
        direction = np.random.choice([-1, 1])
        size = np.random.exponential(0.02)
        self.price += direction * size * self._kyle_lambda()
        self.price = np.clip(self.price, 0.01, 0.99)
        self.volume += size
        self.noise_pnl -= abs(self.price - self.true_prob) * size * 0.5
        self._update_book()
    
    def _mm_update(self):
        spread = max(0.02, 0.05 * (1 - self.volume / 100))
        self.best_bid = self.price - spread / 2
        self.best_ask = self.price + spread / 2
    
    def _kyle_lambda(self):
        sigma_v = abs(self.true_prob - self.price) + 0.05
        sigma_u = 0.1 * np.sqrt(self.n_noise)
        return sigma_v / (2 * sigma_u)
    
    def _update_book(self):
        spread = self.best_ask - self.best_bid
        self.best_bid = self.price - spread / 2
        self.best_ask = self.price + spread / 2
    
    def run(self, n_steps=1000):
        for _ in range(n_steps):
            self.step()
        return np.array(self.price_history)

# ==================== 抓 Polymarket 真實價格 (已強化報錯與 flush) ====================

def get_market_probs(slugs):
    probs = []
    for slug in slugs:
        url = f"https://gamma-api.polymarket.com/markets?slug={slug}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if not data or len(data) == 0:
                print(f"\n❌ [關鍵錯誤] 無法定位市場 slug：'{slug}'", flush=True)
                return None
            
            market = data[0]
            
            # 依序嘗試所有可能的價格欄位
            yes_price = (
                market.get('yes_price') or
                market.get('last_trade_price') or
                market.get('bestAsk') or
                market.get('outcomePrices') and float(market['outcomePrices'][0])  # 多選題
            )
            
            # 最後手段：印出所有欄位讓你 debug
            if yes_price is None:
                available_keys = {k: v for k, v in market.items() if v is not None and 'price' in k.lower()}
                print(f"\n⚠️ [{slug}] 找不到價格，可用的 price 相關欄位：{available_keys}", flush=True)
                print(f"👉 完整欄位：{list(market.keys())}", flush=True)
                return None
                
            probs.append(float(yes_price))
            print(f"✅ 抓到 {slug}：機率 {float(yes_price):.1%}", flush=True)
            
        except Exception as e:
            print(f"\n❌ [系統錯誤] 處理 '{slug}' 時發生異常: {str(e)}", flush=True)
            return None
            
    return np.array(probs)
# ==================== 主程式 ====================
def main():
    parser = argparse.ArgumentParser(description="Polymarket 模擬 bot（文章所有 code 已打包）")
    parser.add_argument('--mode', choices=['binary_demo', 'rare_demo', 'particle_demo', 'stratified_demo', 'correlated_demo', 'abm_demo'],
                        default='correlated_demo', help='要跑哪個模式')
    parser.add_argument('--markets', type=str, default='presidential-winner-2024,will-trump-win',
                        help='correlated_demo 用，逗號分隔的市場 slug')
    parser.add_argument('--copula', choices=['gaussian', 't', 'clayton'], default='t',
                        help='correlated_demo 要用哪種 copula')
    parser.add_argument('--num_sim', type=int, default=100000, help='模擬次數')
    parser.add_argument('--corr_strength', type=float, default=0.6, help='相關性強度 0~1')
    args = parser.parse_args()

    if args.mode == 'binary_demo':
        result = simulate_binary_contract(S0=195, K=200, mu=0.08, sigma=0.20, T=30/365)
        print(f"P(AAPL > $200) ≈ {result['probability']:.4f}")

    elif args.mode == 'rare_demo':
        result = rare_event_IS(S0=5000, K_crash=0.20, sigma=0.15, T=5/252)
        print(f"IS estimate: {result['p_IS']:.6f} ± {result['se_IS']:.6f}")

    elif args.mode == 'particle_demo':
        pf = PredictionMarketParticleFilter(prior_prob=0.50, process_vol=0.03)
        observations = [0.50, 0.52, 0.55, 0.58, 0.61, 0.63, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        print("Election Night Tracker:")
        for t, obs in enumerate(observations):
            pf.update(obs)
            ci = pf.credible_interval()
            print(f"{t:>5}h  {obs:>8.3f}  {pf.estimate():>8.3f}  ({ci[0]:.3f}-{ci[1]:.3f})")

    elif args.mode == 'stratified_demo':
        p, se = stratified_binary_mc(S0=100, K=105, sigma=0.20, T=30/365)
        print(f"Stratified estimate: {p:.6f} ± {se:.6f}")

    elif args.mode == 'correlated_demo':
        # 使用正則表達式提取所有非分隔符號的字串
        slugs = re.findall(r"[\w-]+", args.markets)
        
        if len(slugs) < 2:
            print(f"解析失敗或數量不足（僅找到 {len(slugs)} 個 slug）。輸入內容：{args.markets}")
            return
        
        probs = get_market_probs(slugs)
        
        if probs is None:
            # 這是關鍵：如果抓不到資料，以錯誤碼 1 退出，讓 Streamlit 捕捉
            import sys
            sys.exit(1)

        n = len(probs)
        corr = np.full((n, n), args.corr_strength)
        np.fill_diagonal(corr, 1.0)

        if args.copula == 'gaussian':
            outcomes = simulate_correlated_outcomes_gaussian(probs, corr, args.num_sim)
        elif args.copula == 't':
            outcomes = simulate_correlated_outcomes_t(probs, corr, nu=4, N=args.num_sim)
        else:
            outcomes = simulate_correlated_outcomes_clayton(probs, theta=2.0, N=args.num_sim)

        p_sweep = outcomes.all(axis=1).mean()
        p_lose_all = (~outcomes.astype(bool)).all(axis=1).mean()
        print(f"\n=== {args.copula.upper()} COPULA 結果 ===", flush=True)
        print(f"市場獨立掃全部機率: {np.prod(probs):.4f}", flush=True)
        print(f"模擬掃全部機率:     {p_sweep:.4f}", flush=True)
        print(f"模擬全輸機率:       {p_lose_all:.4f}", flush=True)

    elif args.mode == 'abm_demo':
        sim = PredictionMarketABM(true_prob=0.65)
        prices = sim.run(n_steps=2000)
        print(f"最終價格: {prices[-1]:.4f}  (真實機率 {sim.true_prob})")

if __name__ == "__main__":
    main()