#!/usr/bin/env python3
"""
1_qiskit-gpu.py  —  Quantum Portfolio Optimization on 7x H100 80 GB SXM
Simulator : cuStateVec GPU (CUDA 12)  via  qiskit-aer-gpu
Execution : each quantum method runs on a dedicated H100 in parallel
Usage     : python 1_qiskit-gpu.py
Requires  : pip install qiskit-aer-gpu cuquantum-cu12==24.8.0
"""

import os, csv, time, warnings
import numpy as np
import cvxpy as cp
import scipy.stats as stats_lib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager, set_start_method

warnings.filterwarnings("ignore")

OUT_DIR = "."

# =================================================================
# H100 GPU CONFIGURATION
# =================================================================

N_GPUS = 7

def _check_gpu():
    try:
        from qiskit_aer import AerSimulator
        from qiskit import QuantumCircuit
        sim = AerSimulator(method="statevector", device="GPU", cuStateVec_enable=True)
        qc  = QuantumCircuit(2); qc.h(0); qc.cx(0, 1); qc.measure_all()
        sim.run(qc, shots=8).result()
        return True
    except Exception as exc:
        print(f"  GPU unavailable: {exc}")
        return False

def make_sampler(gpu_id: int = 0, shots: int = 2048):
    """Build an AerSampler pinned to a specific H100 via cuStateVec GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from qiskit_aer.primitives import Sampler as AerSampler
    s = AerSampler()
    if GPU_AVAILABLE:
        s.set_options(
            shots             = shots,
            seed_simulator    = 42,
            device            = "GPU",
            cuStateVec_enable = True,
            precision         = "single",   # H100 FP32 / tensor-core path
        )
    else:
        s.set_options(shots=shots, seed_simulator=42, method="statevector")
    return s

GPU_AVAILABLE = _check_gpu()
print(f"H100 GPU : {'cuStateVec GPU' if GPU_AVAILABLE else 'CPU statevector (fallback)'}")
print(f"GPUs     : {N_GPUS}  (ids 0-{N_GPUS-1})")

# =================================================================
# SECTION 1 — CLASSICAL MARKOWITZ
# =================================================================

# ── Load CSV using standard library (no pandas dependency) ────────
with open(CSV_FILE) as f:
    rows    = list(csv.reader(f))
headers     = rows[0]
TICKERS     = headers[1:]                          # 40 asset names
dates       = [r[0] for r in rows[1:]]
price_data  = np.array([[float(v) for v in r[1:]]
                         for r in rows[1:]], dtype=float)

N = len(TICKERS)    # 40 assets
T = len(dates)      # 2765 trading days

print(f"Assets         : {N}")
print(f"Trading days   : {T}")
print(f"Date range     : {dates[0]}  →  {dates[-1]}")
print(f"Tickers        : {', '.join(TICKERS)}")

# ── Log returns: r_t = ln(P_t / P_{t-1}) ─────────────────────────
log_ret = np.diff(np.log(price_data), axis=0)   # shape (T-1, N)

# ── Annualised statistics ──────────────────────────────────────────
mu    = log_ret.mean(axis=0) * 252               # expected annual return
sigma = log_ret.std(axis=0)  * np.sqrt(252)      # annual volatility  (1D)
cov   = np.cov(log_ret.T)    * 252               # annual covariance (N×N)

print(f"Log returns shape : {log_ret.shape}  (trading days × assets)")
print(f"  μ  range : {mu.min():.4f}  →  {mu.max():.4f}")
print(f"  σ  range : {sigma.min():.4f}  →  {sigma.max():.4f}")
print(f"  Σ  shape : {cov.shape}")

# ── Per-asset statistics ───────────────────────────────────────────
plt.savefig(os.path.join(OUT_DIR, f'plot_{int(time.time())}.png'), dpi=130, bbox_inches='tight'); plt.close()

## QUBO asset scores across lambda values

PRE_SCREEN_N = 15   # pre-screen pool before VQE (must be > BUDGET)
lambdas = [0.1, 0.3, 0.5, 0.7, 0.9]

# σᵢᵢ = variance of asset i = diagonal of covariance matrix (cov is 2D, sigma is 1D std-dev)
scores_05 = np.array([-0.5*mu[i] + 0.5*cov[i,i]/2 for i in range(N)])

print('='*90)
print(f'ASSET QUBO SCORE = −λ·μᵢ + (1−λ)·σᵢᵢ/2   at λ=0.5  (lower = more attractive to VQE)')
print('='*90)
print(f'{"Rank":<6}{"Asset":<8}' + ''.join(f'{"λ="+str(l):<13}' for l in lambdas) + '  Decision@λ=0.5')
print('-'*90)
ranked_idx = np.argsort(scores_05)
for rank, i in enumerate(ranked_idx, 1):
    scores_row = [-l*mu[i] + (1-l)*cov[i,i]/2 for l in lambdas]
    decision = (f'✓ top-{BUDGET}' if rank <= BUDGET
                else ('◉ pre-screen' if rank <= PRE_SCREEN_N else '✗ discarded'))
    print(f'[{rank:2d}]  {TICKERS[i]:<8}' + ''.join(f'{s:+12.6f} ' for s in scores_row) + f'  {decision}')

print()
print('LEGEND:')
print(f'  ✓ top-{BUDGET}      : enters the final portfolio')
print(f'  ◉ pre-screen : passes pre-filter (ranks {BUDGET+1}–{PRE_SCREEN_N}); enters VQE universe')
print(f'  ✗ discarded  : too volatile or too low return at λ=0.5')

# Lambda sweep plot
fig, ax = plt.subplots(figsize=(13, 6))
lam_range = np.linspace(0.05, 0.95, 200)
cmap = plt.cm.tab20(np.linspace(0, 1, N))
for i in range(N):
    sc = [-l*mu[i] + (1-l)*cov[i,i]/2 for l in lam_range]
    ax.plot(lam_range, sc, color=cmap[i], lw=1.2, label=TICKERS[i])
ax.axvline(0.5, color='red', ls='--', lw=2, label='Current λ=0.5')
ax.set_xlabel('Risk Factor λ', fontsize=12)
ax.set_ylabel('Solo-Asset QUBO Score (lower = better)', fontsize=12)
ax.set_title('Effect of λ on All 40 Asset QUBO Scores', fontsize=13)
ax.legend(ncol=5, fontsize=6, loc='upper right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'risk_factor_sweep_40.png'), dpi=130)
plt.savefig(os.path.join(OUT_DIR, f'plot_{int(time.time())}.png'), dpi=130, bbox_inches='tight'); plt.close()

# ── Per-asset Sharpe (individual, not portfolio) ──────────────────
individual_sharpe = (mu - RISK_FREE) / sigma

print(f"\nTop 5 assets by individual Sharpe:")
top5 = np.argsort(individual_sharpe)[-5:][::-1]
for i in top5:
    print(f"  {TICKERS[i]:<8}  μ={mu[i]*100:.2f}%  σ={sigma[i]*100:.2f}%  "
          f"Sharpe={individual_sharpe[i]:.4f}")

# ── Markowitz QP via cvxpy (CLARABEL solver)  ─────────────────────

# WHY  CLARABEL_SOlver(Fastest due to written in RUST)is :
#- OSQP → simple QP, fastest
#- ECOS → QP + SOCP, small problems
#- SCS → QP + SOCP, large/complex problems
#- GLPK_MI → cardinality constraint (binary variables)
#- CLARABEL → all of the above in one solver (that's why it's the default)


# Alternate to cvxpy(used in CLARABEL) is :

# 1) scipy.optimize.minimize — no extra install needed
# 2) quadprog — lightweight pure QP
# 3) PyPortfolioOpt — purpose-built for portfolios
# 4) osqp — what CVXPY uses internally  ----- etc.

## No selection is taking place , 
#on the basis of weightage provided to the assets , decide slectection

w       = cp.Variable(N, nonneg=True)
mkt_obj = LAMBDA * cp.quad_form(w, cov) - (1 - LAMBDA) * mu @ w

prob = cp.Problem(cp.Minimize(mkt_obj), [cp.sum(w) == 1])

t0 = time.perf_counter()
prob.solve(solver=cp.CLARABEL)
solve_time = time.perf_counter() - t0

w_opt = w.value    # continuous optimal weights (N,)

print(f"Solver status  : {prob.status}")
print(f"Objective val  : {prob.value:.6f}")
print(f"Solve time     : {solve_time*1000:.2f} ms")
print(f"\nTop 10 assets by Markowitz weight:")
top10_idx = np.argsort(w_opt)[-BUDGET:][::-1]
for rank, i in enumerate(top10_idx, 1):
    print(f"  {rank:2d}. {TICKERS[i]:<8}  weight={w_opt[i]*100:.2f}%  "
          f"μ={mu[i]*100:.2f}%  σ={sigma[i]*100:.2f}%")

# ── Helper: portfolio stats for a set of asset indices ────────────
def portfolio_stats(asset_idx, weights=None):
    """
    Compute return, risk, Sharpe for a portfolio.
    If weights=None → equal weighting (1/K each).
    """
    idx = np.array(asset_idx)
    k   = len(idx)
    w   = weights if weights is not None else np.ones(k) / k

    ret    = w @ mu[idx]
    var    = w @ cov[np.ix_(idx, idx)] @ w
    risk   = np.sqrt(var)
    sharpe = (ret - RISK_FREE) / risk
    obj    = LAMBDA * var - (1 - LAMBDA) * ret

    return {
        'return' : ret,
        'risk'   : risk,
        'sharpe' : sharpe,
        'obj'    : obj,
        'weights': w,
        'tickers': [TICKERS[i] for i in idx],
    }

# ── Portfolio 1: Markowitz continuous weights ─────────────────────
mkt_continuous = portfolio_stats(np.arange(N), weights=w_opt)

# ── Portfolio 2: Top-K binary selection (equal weight) ───────────
binary_idx    = top10_idx
mkt_binary    = portfolio_stats(binary_idx)

# ── Portfolio 3: Markowitz weights within top-K selection ─────────
top_w         = w_opt[binary_idx]
top_w         = top_w / top_w.sum()   # re-normalise within selection
mkt_top_w     = portfolio_stats(binary_idx, weights=top_w)

# ── Print results ─────────────────────────────────────────────────
print("=" * 72)
print("  MARKOWITZ PORTFOLIO — SHARPE RATIO RESULTS")
print("=" * 72)

rows_data = [
    ("All-40 continuous",   mkt_continuous),
    ("Top-10 equal weight", mkt_binary),
    ("Top-10 MKT weights",  mkt_top_w),
]
print(f"  {'Portfolio':<26} {'Return':>8} {'Risk':>8} {'Sharpe':>9} {'Obj':>10}")
print("-" * 72)
for name, s in rows_data:
    print(f"  {name:<26} {s['return']*100:>7.3f}% {s['risk']*100:>7.3f}% "
          f"{s['sharpe']:>9.4f} {s['obj']:>10.6f}")
print("=" * 72)

print(f"\n  Best Sharpe : {max(rows_data, key=lambda x: x[1]['sharpe'])[0]}")
print(f"  Risk-free   : {RISK_FREE*100:.1f}%")

# ── Monte Carlo: 2000 random portfolios (equal weight) ────────────
rng = np.random.default_rng(42)
mc_ret, mc_risk, mc_sharpe = [], [], []

for _ in range(2000):
    idx = rng.choice(N, BUDGET, replace=False)
    s   = portfolio_stats(idx)
    mc_ret.append(s['return'] * 100)
    mc_risk.append(s['risk']  * 100)
    mc_sharpe.append(s['sharpe'])

mc_ret    = np.array(mc_ret)
mc_risk   = np.array(mc_risk)
mc_sharpe = np.array(mc_sharpe)

print(f"Monte Carlo portfolios : 2000")
print(f"Return range   : [{mc_ret.min():.2f}%, {mc_ret.max():.2f}%]")
print(f"Risk range     : [{mc_risk.min():.2f}%, {mc_risk.max():.2f}%]")
print(f"Sharpe range   : [{mc_sharpe.min():.3f},  {mc_sharpe.max():.3f}]")
print(f"\nMarkowitz (top-10 equal)  Sharpe = {mkt_binary['sharpe']:.4f}")
print(f"Monte Carlo best random portfolio  Sharpe = {mc_sharpe.max():.4f}")
print(f"Markowitz advantage       = {(mkt_binary['sharpe'] - mc_sharpe.mean()):.4f} vs MC mean")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# ── Plot 1: Efficient Frontier ────────────────────────────────────
ax = axes[0, 0]
sc = ax.scatter(mc_risk, mc_ret, c=mc_sharpe, cmap='viridis', s=6, alpha=0.4)
plt.colorbar(sc, ax=ax, label='Sharpe Ratio', fraction=0.04)

markers = [
    (mkt_binary,    '#D85A30', '*', 250, f"Markowitz top-10 equal  (S={mkt_binary['sharpe']:.3f})"),
    (mkt_top_w,     '#534AB7', 'D', 140, f"Markowitz top-10 MKT w  (S={mkt_top_w['sharpe']:.3f})"),
    (mkt_continuous,'#1D9E75', 's', 120, f"Markowitz all-40 contin (S={mkt_continuous['sharpe']:.3f})"),
]
for s, col, mrk, sz, lbl in markers:
    ax.scatter(s['risk']*100, s['return']*100, s=sz, c=col,
               marker=mrk, zorder=7, label=lbl)

ax.set_xlabel('Portfolio Risk (%)', fontsize=11)
ax.set_ylabel('Portfolio Return (%)', fontsize=11)
ax.set_title('MC Efficient Frontier', fontsize=11, fontweight='bold')
ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# ── Plot 2: Asset weights bar chart ──────────────────────────────
ax = axes[0, 1]
tickers_top = [TICKERS[i] for i in top10_idx]
w_pct       = top_w * 100
bars = ax.barh(tickers_top, w_pct, color='#378ADD', alpha=0.85)
for bar, v in zip(bars, w_pct):
    ax.text(v + 0.1, bar.get_y() + bar.get_height()/2,
            f'{v:.1f}%', va='center', fontsize=9)
ax.set_xlabel('Markowitz Weight (%)', fontsize=11)
ax.set_title('Top-10 Portfolio — Markowitz Weights', fontsize=11, fontweight='bold')
ax.grid(True, axis='x', alpha=0.2)

# ── Plot 3: Individual asset Sharpe ──────────────────────────────
ax = axes[1, 0]
sorted_idx = np.argsort(individual_sharpe)
colors_bar = ['#D85A30' if i in top10_idx else '#AAAAAA' for i in sorted_idx]
ax.barh([TICKERS[i] for i in sorted_idx], individual_sharpe[sorted_idx],
        color=colors_bar, alpha=0.85)
ax.axvline(0, color='black', lw=0.8, ls='--')
ax.set_xlabel('Individual Sharpe Ratio', fontsize=11)
ax.set_title('Per-Asset Sharpe (orange = selected)', fontsize=11, fontweight='bold')
ax.grid(True, axis='x', alpha=0.2)

# ── Plot 4: Sharpe comparison bar ────────────────────────────────
ax = axes[1, 1]
labels_cmp = ['MC Mean\n(random)', 'MC Best\n(random)',
               'Markowitz\nall-40', 'Markowitz\ntop-10 eq.',
               'Markowitz\ntop-10 MKT w']
sharpe_cmp = [mc_sharpe.mean(), mc_sharpe.max(),
              mkt_continuous['sharpe'], mkt_binary['sharpe'], mkt_top_w['sharpe']]
cols_cmp   = ['#AAAAAA','#CCCCCC','#1D9E75','#D85A30','#534AB7']
b = ax.bar(labels_cmp, sharpe_cmp, color=cols_cmp, alpha=0.85, width=0.6)
for bar, v in zip(b, sharpe_cmp):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.005,
            f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Sharpe Ratio', fontsize=11)
ax.set_title('Sharpe Ratio Comparison', fontsize=11, fontweight='bold')
ax.set_ylim(bottom=min(sharpe_cmp)*0.95)
ax.grid(True, axis='y', alpha=0.2)
ax.tick_params(axis='x', labelsize=8)

plt.suptitle('Classical Markowitz Portfolio Optimization\n'
             f'40 Assets · 2015–2025 · Budget K={BUDGET} · λ={LAMBDA}',
             fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f'plot_{int(time.time())}.png'), dpi=130, bbox_inches='tight'); plt.close()

# ── Full Summary ──────────────────────────────────────────────────
print("=" * 70)
print("  CLASSICAL MARKOWITZ — FINAL SUMMARY")
print(f"  Dataset  : {N} assets · {T} days · {dates[0]} → {dates[-1]}")
print(f"  λ={LAMBDA}  K={BUDGET}  Risk-free={RISK_FREE*100:.1f}%")
print("=" * 70)

print(f"\n  OPTIMAL PORTFOLIO  (top-{BUDGET} by Markowitz weight, equal-weight eval)")
print(f"  Assets   : {', '.join(mkt_binary['tickers'])}")
print(f"  Return   : {mkt_binary['return']*100:.3f}%  (annualised)")
print(f"  Risk     : {mkt_binary['risk']*100:.3f}%  (annualised std-dev)")
print(f"  Sharpe   : {mkt_binary['sharpe']:.4f}")
print(f"  Obj H    : {mkt_binary['obj']:.6f}")

print(f"\n  HOW IT RANKS:")
print(f"  MC mean Sharpe  : {mc_sharpe.mean():.4f}  (avg random portfolio)")
print(f"  MC best Sharpe  : {mc_sharpe.max():.4f}  (best of 2000 random)")
print(f"  Markowitz Sharpe: {mkt_binary['sharpe']:.4f}  ← optimised")
pct = np.mean(mc_sharpe < mkt_binary['sharpe']) * 100
print(f"  Beats {pct:.1f}% of all random portfolios")

print(f"\n  SHARPE FORMULA:")
print(f"  Sharpe = (Return − Risk-free) / Risk")
print(f"         = ({mkt_binary['return']*100:.3f}% − {RISK_FREE*100:.1f}%) / {mkt_binary['risk']*100:.3f}%")
print(f"         = {mkt_binary['return']-RISK_FREE:.4f} / {mkt_binary['risk']:.4f}")
print(f"         = {mkt_binary['sharpe']:.4f}")
print("=" * 70)

# =================================================================
# QUANTUM HELPERS  (shared globals — built once in main process)
# =================================================================

from qiskit.circuit.library import TwoLocal, QAOAAnsatz
from qiskit_algorithms import SamplingVQE
from qiskit_algorithms.optimizers import COBYLA, Optimizer, OptimizerResult
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import to_ising

# ── Quantum constants ──────────────────────────────────────────────────
N_Q        = 12      # quantum candidate pool  (> BUDGET)
BUDGET_Q   = 8         # assets selected by quantum solver
LAMBDA_Q   = 0.5     # risk-return trade-off
SHOTS_OPT  = 2048    # shots inside optimizer loop
SHOTS_DEC  = 4096    # shots for final decode
ALPHA_CVAR = 0.1     # CVaR tail (worst 10%)
REPS       = 2         # QAOA / VQE repetitions
BETA_DRO   = 0.05    # DRO confidence (95%)
T_OBS      = log_ret.shape[0]

# ── Pre-select top N_Q by Markowitz QP weight ─────────────────────────
q_sub_idx   = np.argsort(w_opt)[-N_Q:][::-1]
sub_mu      = mu[q_sub_idx]
sub_cov     = cov[np.ix_(q_sub_idx, q_sub_idx)]
sub_tickers = [TICKERS[i] for i in q_sub_idx]
print(f"Quantum pool ({N_Q} assets): {sub_tickers}")

# ── Hamiltonian builder ────────────────────────────────────────────────
def make_operator(s_mu, s_cov):
    po   = PortfolioOptimization(expected_returns=s_mu, covariances=s_cov,
                                 risk_factor=LAMBDA_Q, budget=BUDGET_Q)
    qubo = QuadraticProgramToQubo().convert(po.to_quadratic_program())
    return to_ising(qubo)   # (SparsePauliOp, float offset)

operator_std, offset_std = make_operator(sub_mu, sub_cov)
print(f"Hamiltonian: {operator_std.num_qubits} qubits, {len(operator_std)} Pauli terms")

# ── Decode quasi-distribution → portfolio ──────────────────────────────
def decode(eigenstate, n, sub_idx, budget):
    if hasattr(eigenstate, 'binary_probabilities'):
        dist = {int(k, 2): v for k, v in eigenstate.binary_probabilities().items()}
    else:
        dist = dict(eigenstate)
    for state_int, _ in sorted(dist.items(), key=lambda x: -x[1]):
        bits = format(state_int, f'0{n}b')[::-1]
        sel  = [sub_idx[i] for i, b in enumerate(bits) if b == '1']
        if len(sel) == budget:
            return np.array(sel)
    return np.array(sub_idx[:budget])   # fallback

# ── Equal-weight portfolio stats ───────────────────────────────────────
def q_stats(idx):
    """Portfolio stats using Markowitz-optimal weights for selected subset."""
    idx      = np.array(idx)
    k        = len(idx)
    s_mu     = mu[idx]
    s_cov    = cov[np.ix_(idx, idx)]
    # Markowitz QP on the selected subset
    w_var    = cp.Variable(k, nonneg=True)
    prob_sub = cp.Problem(
        cp.Minimize(LAMBDA_Q * cp.quad_form(w_var, s_cov) - (1-LAMBDA_Q) * s_mu @ w_var),
        [cp.sum(w_var) == 1]
    )
    prob_sub.solve(solver=cp.CLARABEL, verbose=False)
    w = w_var.value if prob_sub.status == 'optimal' and w_var.value is not None         else np.ones(k) / k
    r = float(w @ s_mu)
    v = float(w @ s_cov @ w)
    return {'return': r, 'risk': np.sqrt(v),
            'sharpe': (r - RISK_FREE) / np.sqrt(v),
            'weights': w,
            'tickers': [TICKERS[i] for i in idx]}

# ── Shared sampler ─────────────────────────────────────────────────────
print("Setup complete.")

sampler = make_sampler(gpu_id=0, shots=SHOTS_OPT)  # default sampler
print("Quantum helpers ready.")


# =================================================================
# QUANTUM METHOD FUNCTIONS  (each pinned to a dedicated H100)
# =================================================================

def run_method_1(result_d):
    """Method 1 — H100 GPU 0."""
    import os, time, numpy as np, cvxpy as cp
    import scipy.stats as stats_lib
    from scipy.optimize import dual_annealing, minimize as sp_min
    from qiskit.circuit.library import TwoLocal, QAOAAnsatz
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer, OptimizerResult
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    sampler = make_sampler(gpu_id=0, shots=SHOTS_OPT)

    ansatz_vqe = TwoLocal(N_Q, 'ry', 'cx', reps=REPS, entanglement='linear')
    print(f"VQE ansatz: {ansatz_vqe.num_parameters} parameters, depth={ansatz_vqe.decompose().depth()}")

    vqe_m1 = SamplingVQE(sampler=sampler, ansatz=ansatz_vqe, optimizer=COBYLA(maxiter=300))

    t0 = time.perf_counter()
    res_m1 = vqe_m1.compute_minimum_eigenvalue(operator_std)
    t_m1   = time.perf_counter() - t0

    sel_m1   = decode(res_m1.eigenstate, N_Q, q_sub_idx, BUDGET_Q)
    stats_m1 = q_stats(sel_m1)
    ip_m1    = res_m1.optimal_point   # save for warm-starting Method 3

    print(f"\nMethod 1 — VQE only")
    print(f"  Energy    : {res_m1.eigenvalue:.6f}")
    print(f"  Wall time : {t_m1:.2f}s  |  Iterations: {res_m1.cost_function_evals}")
    print(f"  Selected  : {stats_m1['tickers']}")
    print(f"  Return    : {stats_m1['return']*100:.2f}%  Risk: {stats_m1['risk']*100:.2f}%  Sharpe: {stats_m1['sharpe']:.4f}")

    # store results
    for _k in list(locals().keys()):
        if _k.startswith(("stats_m","t_m","t_ws","t_m8")):
            result_d[_k] = locals()[_k]
    for _k in list(locals().keys()):
        if _k.startswith("res_m"):
            try: result_d[_k.replace("res_","ev_")] = float(locals()[_k].eigenvalue)
            except: pass
def run_method_2(result_d):
    """Method 2 — H100 GPU 1."""
    import os, time, numpy as np, cvxpy as cp
    import scipy.stats as stats_lib
    from scipy.optimize import dual_annealing, minimize as sp_min
    from qiskit.circuit.library import TwoLocal, QAOAAnsatz
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer, OptimizerResult
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
    sampler = make_sampler(gpu_id=1, shots=SHOTS_OPT)

    ansatz_qaoa = QAOAAnsatz(operator_std, reps=REPS)
    print(f"QAOA ansatz: {ansatz_qaoa.num_parameters} parameters (beta + gamma per rep)")

    vqe_m2 = SamplingVQE(sampler=sampler, ansatz=ansatz_qaoa,
                         optimizer=COBYLA(maxiter=300),
                         initial_point=np.zeros(ansatz_qaoa.num_parameters))

    t0 = time.perf_counter()
    res_m2 = vqe_m2.compute_minimum_eigenvalue(operator_std)
    t_m2   = time.perf_counter() - t0

    sel_m2   = decode(res_m2.eigenstate, N_Q, q_sub_idx, BUDGET_Q)
    stats_m2 = q_stats(sel_m2)

    print(f"\nMethod 2 — QAOA only")
    print(f"  Energy    : {res_m2.eigenvalue:.6f}")
    print(f"  Wall time : {t_m2:.2f}s  |  Iterations: {res_m2.cost_function_evals}")
    print(f"  Selected  : {stats_m2['tickers']}")
    print(f"  Return    : {stats_m2['return']*100:.2f}%  Risk: {stats_m2['risk']*100:.2f}%  Sharpe: {stats_m2['sharpe']:.4f}")

    # store results
    for _k in list(locals().keys()):
        if _k.startswith(("stats_m","t_m","t_ws","t_m8")):
            result_d[_k] = locals()[_k]
    for _k in list(locals().keys()):
        if _k.startswith("res_m"):
            try: result_d[_k.replace("res_","ev_")] = float(locals()[_k].eigenvalue)
            except: pass
def run_method_3(result_d):
    """Method 3 — H100 GPU 2."""
    import os, time, numpy as np, cvxpy as cp
    import scipy.stats as stats_lib
    from scipy.optimize import dual_annealing, minimize as sp_min
    from qiskit.circuit.library import TwoLocal, QAOAAnsatz
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer, OptimizerResult
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
    sampler = make_sampler(gpu_id=2, shots=SHOTS_OPT)

    # Warm-start: derive QAOA initial β, γ from VQE result
    # NOTE: This is a heuristic mapping, not a derived formula.
    # γ = -E_vqe / (2·N_Q) scales by VQE energy; β = π/4 is a common empirical choice.
    # No theoretical guarantee — used as an informed starting point only.
    gamma_ws = float(np.clip(-res_m1.eigenvalue / (2 * N_Q), -np.pi, np.pi))
    beta_ws  = np.pi / 4
    qaoa_ip_ws = np.array([beta_ws] * REPS + [gamma_ws] * REPS)
    print(f"VQE energy={res_m1.eigenvalue:.4f}  →  QAOA init β={beta_ws:.3f}, γ={gamma_ws:.3f}")

    ansatz_m3 = QAOAAnsatz(operator_std, reps=REPS)
    vqe_m3 = SamplingVQE(sampler=sampler, ansatz=ansatz_m3,
                         optimizer=COBYLA(maxiter=300),
                         initial_point=qaoa_ip_ws)

    t0 = time.perf_counter()
    res_m3 = vqe_m3.compute_minimum_eigenvalue(operator_std)
    t_m3   = time.perf_counter() - t0

    sel_m3   = decode(res_m3.eigenstate, N_Q, q_sub_idx, BUDGET_Q)
    stats_m3 = q_stats(sel_m3)

    print(f"\nMethod 3 — VQE + QAOA")
    print(f"  Energy    : {res_m3.eigenvalue:.6f}  (QAOA only was {res_m2.eigenvalue:.6f})")
    print(f"  Wall time : {t_m3:.2f}s  |  Iterations: {res_m3.cost_function_evals}")
    print(f"  Selected  : {stats_m3['tickers']}")
    print(f"  Return    : {stats_m3['return']*100:.2f}%  Risk: {stats_m3['risk']*100:.2f}%  Sharpe: {stats_m3['sharpe']:.4f}")

    # store results
    for _k in list(locals().keys()):
        if _k.startswith(("stats_m","t_m","t_ws","t_m8")):
            result_d[_k] = locals()[_k]
    for _k in list(locals().keys()):
        if _k.startswith("res_m"):
            try: result_d[_k.replace("res_","ev_")] = float(locals()[_k].eigenvalue)
            except: pass
def run_method_4(result_d):
    """Method 4 — H100 GPU 3."""
    import os, time, numpy as np, cvxpy as cp
    import scipy.stats as stats_lib
    from scipy.optimize import dual_annealing, minimize as sp_min
    from qiskit.circuit.library import TwoLocal, QAOAAnsatz
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer, OptimizerResult
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(3)
    sampler = make_sampler(gpu_id=3, shots=SHOTS_OPT)

    # ── Delage-Ye γ calibration ──────────────────────────────────────────
    n_sub   = N_Q
    GAMMA_1 = (np.sqrt(n_sub) + np.sqrt(2*np.log(1/BETA_DRO)))**2 / T_OBS
    GAMMA_2 = 1 + np.sqrt((n_sub + 2*np.sqrt(n_sub*np.log(2/BETA_DRO))
                            + 2*np.log(2/BETA_DRO)) / (T_OBS - 1))
    print(f"DRO: γ₁={GAMMA_1:.5f}  γ₂={GAMMA_2:.5f}  (T={T_OBS}, n={n_sub}, β={BETA_DRO})")

    # ── SOCP: find DY-optimal weights ────────────────────────────────────
    Sigma_half = np.linalg.cholesky(sub_cov + 1e-8*np.eye(n_sub))
    w_dy = cp.Variable(n_sub, nonneg=True)
    t_dy = cp.Variable(nonneg=True)
    dy_prob = cp.Problem(
        cp.Minimize(LAMBDA_Q*GAMMA_2*cp.quad_form(w_dy, sub_cov)
                    - (1-LAMBDA_Q)*sub_mu@w_dy
                    + (1-LAMBDA_Q)*np.sqrt(GAMMA_1)*t_dy),
        [cp.sum(w_dy)==1, cp.norm(Sigma_half@w_dy, 2)<=t_dy, w_dy<=1]
    )
    dy_prob.solve(solver=cp.CLARABEL, verbose=False)
    w_dy_opt = w_dy.value

    # ── Worst-case moment parameters ──────────────────────────────────────
    sw       = Sigma_half @ w_dy_opt
    mu_DY    = sub_mu - np.sqrt(GAMMA_1) * (sub_cov @ w_dy_opt) / np.linalg.norm(sw)
    Sigma_DY = GAMMA_2 * sub_cov
    print(f"Worst-case μ* range : {mu_DY.min():.4f} → {mu_DY.max():.4f}  (original: {sub_mu.min():.4f} → {sub_mu.max():.4f})")

    # ── DRO Hamiltonian ───────────────────────────────────────────────────
    operator_dy, offset_dy = make_operator(mu_DY, Sigma_DY)

    # ── QAOA with CVaR on DRO Hamiltonian ────────────────────────────────
    ansatz_m4 = QAOAAnsatz(operator_dy, reps=REPS)
    vqe_m4 = SamplingVQE(sampler=sampler, ansatz=ansatz_m4,
                         optimizer=COBYLA(maxiter=300),
                         initial_point=qaoa_ip_ws,
                         aggregation=ALPHA_CVAR)     # CVaR 10%

    t0 = time.perf_counter()
    res_m4 = vqe_m4.compute_minimum_eigenvalue(operator_dy)
    t_m4   = time.perf_counter() - t0

    sel_m4   = decode(res_m4.eigenstate, N_Q, q_sub_idx, BUDGET_Q)
    stats_m4 = q_stats(sel_m4)

    print(f"\nMethod 4 — VQE + QAOA + DRO-CVaR (α={ALPHA_CVAR})")
    print(f"  Energy    : {res_m4.eigenvalue:.6f}")
    print(f"  Wall time : {t_m4:.2f}s  |  Iterations: {res_m4.cost_function_evals}")
    print(f"  Selected  : {stats_m4['tickers']}")
    print(f"  Return    : {stats_m4['return']*100:.2f}%  Risk: {stats_m4['risk']*100:.2f}%  Sharpe: {stats_m4['sharpe']:.4f}")

    # store results
    for _k in list(locals().keys()):
        if _k.startswith(("stats_m","t_m","t_ws","t_m8")):
            result_d[_k] = locals()[_k]
    for _k in list(locals().keys()):
        if _k.startswith("res_m"):
            try: result_d[_k.replace("res_","ev_")] = float(locals()[_k].eigenvalue)
            except: pass
def run_method_5(result_d):
    """Method 5 — H100 GPU 4."""
    import os, time, numpy as np, cvxpy as cp
    import scipy.stats as stats_lib
    from scipy.optimize import dual_annealing, minimize as sp_min
    from qiskit.circuit.library import TwoLocal, QAOAAnsatz
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer, OptimizerResult
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(4)
    sampler = make_sampler(gpu_id=4, shots=SHOTS_OPT)


    class CMAESOptimizer(Optimizer):
        """CMA-ES wrapper for qiskit_algorithms. Falls back to differential_evolution."""
        def __init__(self, sigma0=0.5, maxiter=200):
            super().__init__()
            self._sigma0  = sigma0
            self._maxiter = maxiter

        @property
        def settings(self):
            return {'sigma0': self._sigma0, 'maxiter': self._maxiter}

        def get_support_level(self):
            from qiskit_algorithms.optimizers import OptimizerSupportLevel
            return {
                'gradient':      OptimizerSupportLevel.ignored,
                'bounds':        OptimizerSupportLevel.ignored,
                'initial_point': OptimizerSupportLevel.supported,
            }

        def minimize(self, fun, x0, jac=None, bounds=None):
            try:
                import cma
                es   = cma.CMAEvolutionStrategy(x0, self._sigma0,
                           {'maxiter': self._maxiter, 'verbose': -9, 'seed': 42})
                nfev = 0
                while not es.stop():
                    xs = es.ask()
                    fs = [float(fun(x)) for x in xs]
                    es.tell(xs, fs)
                    nfev += len(xs)
                r = OptimizerResult()
                r.x = np.array(es.result.xbest)
                r.fun = float(es.result.fbest)
                r.nfev = nfev
                print(f"  CMA-ES: {nfev} evaluations")
                return r
            except ImportError:
                from scipy.optimize import differential_evolution
                bds = [(-np.pi, np.pi)] * len(x0) if bounds is None else bounds
                res = differential_evolution(fun, bds, maxiter=self._maxiter, seed=42, tol=1e-6)
                r = OptimizerResult()
                r.x = res.x; r.fun = float(res.fun); r.nfev = res.nfev
                print(f"  DiffEvol fallback: {res.nfev} evaluations")
                return r

    ansatz_m5 = QAOAAnsatz(operator_dy, reps=REPS)
    vqe_m5 = SamplingVQE(sampler=sampler, ansatz=ansatz_m5,
                         optimizer=CMAESOptimizer(sigma0=0.5, maxiter=150),
                         initial_point=qaoa_ip_ws,
                         aggregation=ALPHA_CVAR)

    t0 = time.perf_counter()
    res_m5 = vqe_m5.compute_minimum_eigenvalue(operator_dy)
    t_m5   = time.perf_counter() - t0

    sel_m5   = decode(res_m5.eigenstate, N_Q, q_sub_idx, BUDGET_Q)
    stats_m5 = q_stats(sel_m5)

    print(f"\nMethod 5 — VQE + QAOA + DRO-CVaR + CMA-ES")
    print(f"  Energy    : {res_m5.eigenvalue:.6f}  (Method 4 was {res_m4.eigenvalue:.6f})")
    print(f"  Wall time : {t_m5:.2f}s")
    print(f"  Selected  : {stats_m5['tickers']}")
    print(f"  Return    : {stats_m5['return']*100:.2f}%  Risk: {stats_m5['risk']*100:.2f}%  Sharpe: {stats_m5['sharpe']:.4f}")

    # store results
    for _k in list(locals().keys()):
        if _k.startswith(("stats_m","t_m","t_ws","t_m8")):
            result_d[_k] = locals()[_k]
    for _k in list(locals().keys()):
        if _k.startswith("res_m"):
            try: result_d[_k.replace("res_","ev_")] = float(locals()[_k].eigenvalue)
            except: pass
def run_method_6(result_d):
    """Method 6 — H100 GPU 5."""
    import os, time, numpy as np, cvxpy as cp
    import scipy.stats as stats_lib
    from scipy.optimize import dual_annealing, minimize as sp_min
    from qiskit.circuit.library import TwoLocal, QAOAAnsatz
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer, OptimizerResult
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(5)
    sampler = make_sampler(gpu_id=5, shots=SHOTS_OPT)

    class TwoPhaseOptimizer(Optimizer):
        """
        Two-Phase Hybrid Optimizer (DA + COBYLA):
          Phase 1 — scipy.dual_annealing (global, avoids local minima)
          Phase 2 — COBYLA (local refinement from Phase 1 result)
        Takes the best of both phases.
        """
        def __init__(self, maxiter_p1=300, maxiter_p2=200):
            super().__init__()
            self._maxiter_p1 = maxiter_p1
            self._maxiter_p2 = maxiter_p2

        @property
        def settings(self):
            return {'maxiter_p1': self._maxiter_p1, 'maxiter_p2': self._maxiter_p2}

        def get_support_level(self):
            from qiskit_algorithms.optimizers import OptimizerSupportLevel
            return {
                'gradient':      OptimizerSupportLevel.ignored,
                'bounds':        OptimizerSupportLevel.ignored,
                'initial_point': OptimizerSupportLevel.supported,
            }

        def minimize(self, fun, x0, jac=None, bounds=None):
            from scipy.optimize import dual_annealing, minimize as sp_min

            bds = [(float(b[0]) if b[0] is not None else -np.pi,
                    float(b[1]) if b[1] is not None else  np.pi)
                   for b in bounds] if bounds is not None else [(-np.pi, np.pi)] * len(x0)

            # Phase 1: Dual Annealing (global)
            print(f"    Phase 1 · Dual Annealing (maxiter={self._maxiter_p1}) ...", end='', flush=True)
            res1 = dual_annealing(fun, bds, x0=x0, maxiter=self._maxiter_p1, seed=42,
                                  minimizer_kwargs={'method': 'COBYLA'})
            print(f" E={res1.fun:.5f}  nfev={res1.nfev}")

            # Phase 2: COBYLA (local refinement)
            print(f"    Phase 2 · COBYLA (maxiter={self._maxiter_p2}) ...", end='', flush=True)
            res2 = sp_min(fun, res1.x, method='COBYLA',
                          options={'maxiter': self._maxiter_p2, 'rhobeg': 0.1})
            print(f" E={res2.fun:.5f}  nfev={res2.nfev}")

            # Take best of both phases
            if res2.fun <= res1.fun:
                best_x, best_fun = res2.x, float(res2.fun)
                print(f"    → Phase 2 (COBYLA) wins: E={best_fun:.5f}")
            else:
                best_x, best_fun = res1.x, float(res1.fun)
                print(f"    → Phase 1 (Dual Annealing) wins: E={best_fun:.5f}")

            r = OptimizerResult()
            r.x   = np.array(best_x)
            r.fun  = best_fun
            r.nfev = res1.nfev + res2.nfev
            return r


    # Step 1: VQE warm-start on DY-DRO Hamiltonian
    print("Step 1 — VQE warm-start on DY-DRO Hamiltonian ...")
    ansatz_ws = TwoLocal(N_Q, 'ry', 'cx', reps=REPS, entanglement='linear')
    vqe_ws    = SamplingVQE(sampler=sampler, ansatz=ansatz_ws, optimizer=COBYLA(maxiter=200))

    t0_ws  = time.perf_counter()
    res_ws = vqe_ws.compute_minimum_eigenvalue(operator_dy)
    t_ws   = time.perf_counter() - t0_ws
    print(f"  VQE on DY-DRO: E={res_ws.eigenvalue:.5f}  ({t_ws:.1f}s)")

    gamma_m6   = float(np.clip(-res_ws.eigenvalue / (2 * N_Q), -np.pi, np.pi))
    beta_m6    = np.pi / 4
    qaoa_ip_m6 = np.array([beta_m6] * REPS + [gamma_m6] * REPS)
    print(f"  QAOA warm-start: β={beta_m6:.3f}  γ={gamma_m6:.3f}")

    # Step 2: QAOA + SPA on DY-DRO Hamiltonian
    print("\nStep 2 — QAOA + SPA (Dual Annealing → COBYLA) ...")
    ansatz_m6 = QAOAAnsatz(operator_dy, reps=REPS)
    vqe_m6 = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz_m6,
        optimizer=TwoPhaseOptimizer(maxiter_p1=300, maxiter_p2=200),
        initial_point=qaoa_ip_m6,
        aggregation=ALPHA_CVAR
    )

    t0 = time.perf_counter()
    res_m6 = vqe_m6.compute_minimum_eigenvalue(operator_dy)
    t_m6   = time.perf_counter() - t0

    sel_m6   = decode(res_m6.eigenstate, N_Q, q_sub_idx, BUDGET_Q)
    stats_m6 = q_stats(sel_m6)

    print(f"\nMethod 6 — QAOA + SPA + VQE warm-start + DY-DRO")
    print(f"  Final energy : {res_m6.eigenvalue:.6f}")
    print(f"  Total nfev   : {res_m6.cost_function_evals}")
    print(f"  Wall time    : {t_ws:.1f}s (VQE) + {t_m6:.1f}s (SPA) = {t_ws+t_m6:.1f}s total")
    print(f"  Selected     : {stats_m6['tickers']}")
    print(f"  Return       : {stats_m6['return']*100:.2f}%  Risk: {stats_m6['risk']*100:.2f}%  Sharpe: {stats_m6['sharpe']:.4f}")

    # store results
    for _k in list(locals().keys()):
        if _k.startswith(("stats_m","t_m","t_ws","t_m8")):
            result_d[_k] = locals()[_k]
    for _k in list(locals().keys()):
        if _k.startswith("res_m"):
            try: result_d[_k.replace("res_","ev_")] = float(locals()[_k].eigenvalue)
            except: pass
def run_method_7(result_d):
    """Method 7 — H100 GPU 6."""
    import os, time, numpy as np, cvxpy as cp
    import scipy.stats as stats_lib
    from scipy.optimize import dual_annealing, minimize as sp_min
    from qiskit.circuit.library import TwoLocal, QAOAAnsatz
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer, OptimizerResult
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(6)
    sampler = make_sampler(gpu_id=6, shots=SHOTS_OPT)


    # ── VQE warm-start on DY-DRO Hamiltonian ─────────────────────────────
    # (reuse operator_dy and offset_dy from Method 4)
    print("Step 1 — VQE warm-start on DY-DRO Hamiltonian ...")
    ansatz_ws7 = TwoLocal(N_Q, 'ry', 'cx', reps=REPS, entanglement='linear')
    vqe_ws7    = SamplingVQE(sampler=sampler, ansatz=ansatz_ws7,
                             optimizer=COBYLA(maxiter=200))

    t0_ws7  = time.perf_counter()
    res_ws7 = vqe_ws7.compute_minimum_eigenvalue(operator_dy)
    t_ws7   = time.perf_counter() - t0_ws7
    print(f"  VQE on DY-DRO: E={res_ws7.eigenvalue:.5f}  ({t_ws7:.1f}s)")

    gamma_m7   = float(np.clip(-res_ws7.eigenvalue / (2 * N_Q), -np.pi, np.pi))
    beta_m7    = np.pi / 4
    qaoa_ip_m7 = np.array([beta_m7] * REPS + [gamma_m7] * REPS)
    print(f"  QAOA warm-start: β={beta_m7:.3f}  γ={gamma_m7:.3f}")

    # ── QAOA + SPSA on DY-DRO Hamiltonian ────────────────────────────────
    # SPSA requires initial_point — use VQE warm-start
    print("\nStep 2 — QAOA + SPSA on DY-DRO Hamiltonian ...")
    spsa = SPSA(
        maxiter=300,
        learning_rate=0.1,       # a: step size scale
        perturbation=0.05,
            seed=42,       # c: perturbation magnitude
    )

    ansatz_m7 = QAOAAnsatz(operator_dy, reps=REPS)
    vqe_m7 = SamplingVQE(
        sampler=sampler,
        ansatz=ansatz_m7,
        optimizer=spsa,
        initial_point=qaoa_ip_m7,    # SPSA requires initial_point
        aggregation=ALPHA_CVAR        # CVaR 10%
    )

    t0 = time.perf_counter()
    res_m7 = vqe_m7.compute_minimum_eigenvalue(operator_dy)
    t_m7   = time.perf_counter() - t0

    sel_m7   = decode(res_m7.eigenstate, N_Q, q_sub_idx, BUDGET_Q)
    stats_m7 = q_stats(sel_m7)

    print(f"\nMethod 7 — QAOA + SPSA + VQE warm-start + DY-DRO")
    print(f"  Final energy : {res_m7.eigenvalue:.6f}")
    print(f"  Iterations   : {res_m7.cost_function_evals}  "
          f"(SPSA uses 2 circuit evals/iter → ~{res_m7.cost_function_evals//2} grad steps)")
    print(f"  Wall time    : {t_ws7:.1f}s (VQE) + {t_m7:.1f}s (SPSA) = {t_ws7+t_m7:.1f}s total")
    print(f"  Selected     : {stats_m7['tickers']}")
    print(f"  Return       : {stats_m7['return']*100:.2f}%  "
          f"Risk: {stats_m7['risk']*100:.2f}%  "
          f"Sharpe: {stats_m7['sharpe']:.4f}")

    # store results
    for _k in list(locals().keys()):
        if _k.startswith(("stats_m","t_m","t_ws","t_m8")):
            result_d[_k] = locals()[_k]
    for _k in list(locals().keys()):
        if _k.startswith("res_m"):
            try: result_d[_k.replace("res_","ev_")] = float(locals()[_k].eigenvalue)
            except: pass
def run_method_8(result_d):
    """Method 8 — H100 GPU 0."""
    import os, time, numpy as np, cvxpy as cp
    import scipy.stats as stats_lib
    from scipy.optimize import dual_annealing, minimize as sp_min
    from qiskit.circuit.library import TwoLocal, QAOAAnsatz
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit_algorithms import SamplingVQE
    from qiskit_algorithms.optimizers import COBYLA, SPSA, Optimizer, OptimizerResult
    from qiskit_finance.applications.optimization import PortfolioOptimization
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import to_ising
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
    sampler = make_sampler(gpu_id=0, shots=SHOTS_OPT)


    print("=" * 65)
    print("METHOD 8 — 3-STAGE PIPELINE")
    print("=" * 65)

    # ══════════════════════════════════════════════════════════════
    # STAGE 1 — ASSET SELECTION  (QAOA + SPSA + DY-DRO Hamiltonian)
    # ══════════════════════════════════════════════════════════════
    print("\nSTAGE 1: Asset selection via QAOA + SPSA")
    print("-" * 45)

    spsa_sel   = SPSA(maxiter=300, learning_rate=0.1, perturbation=0.05, seed=42)
    ansatz_sel = QAOAAnsatz(operator_dy, reps=REPS)

    # VQE warm-start for QAOA initial β/γ (run on DY-DRO Hamiltonian)
    vqe_init = SamplingVQE(
        sampler=sampler,
        ansatz=TwoLocal(N_Q, 'ry', 'cx', reps=1, entanglement='linear'),
        optimizer=COBYLA(maxiter=150)
    )
    res_init   = vqe_init.compute_minimum_eigenvalue(operator_dy)
    gamma_s1   = float(np.clip(-res_init.eigenvalue / (2 * N_Q), -np.pi, np.pi))
    ip_s1      = np.array([np.pi / 4] * REPS + [gamma_s1] * REPS)

    vqe_sel = SamplingVQE(
        sampler=sampler, ansatz=ansatz_sel,
        optimizer=spsa_sel, initial_point=ip_s1,
        aggregation=ALPHA_CVAR
    )
    t0 = time.perf_counter()
    res_sel = vqe_sel.compute_minimum_eigenvalue(operator_dy)
    t_s1    = time.perf_counter() - t0

    selected  = decode(res_sel.eigenstate, N_Q, q_sub_idx, BUDGET_Q)
    sel_names = [TICKERS[i] for i in selected]
    print(f"  Selection energy : {res_sel.eigenvalue:.5f}")
    print(f"  Wall time        : {t_s1:.1f}s")
    print(f"  Selected assets  : {sel_names}")

    # Positions of selected assets within the N_Q sub-space
    sel_pos    = [int(np.where(q_sub_idx == idx)[0][0]) for idx in selected]
    K          = len(selected)

    # DRO sub-parameters for the selected K assets
    mu_sel_dy  = mu_DY[sel_pos]           # worst-case returns
    cov_sel_dy = Sigma_DY[np.ix_(sel_pos, sel_pos)]  # worst-case covariance
    mu_sel_std = sub_mu[sel_pos]          # historical returns (for comparison)
    cov_sel_std = sub_cov[np.ix_(sel_pos, sel_pos)]

    # ══════════════════════════════════════════════════════════════
    # STAGE 2 — WEIGHT ALLOCATION  (Quantum-Parameterized Weight Optimizer)
    # ══════════════════════════════════════════════════════════════
    print(f"\nSTAGE 2: Quantum-Parameterized Weight Optimizer (Ry circuit + DA + COBYLA)")
    print("-" * 45)

    # ── Quantum-parameterized weight circuit ────────────────────────
    # This is NOT standard VQE (which minimises ⟨ψ|H|ψ⟩ for a quantum H).
    # Instead: Ry circuit parameterises weights via P(|1⟩) ∝ weight,
    # and a classical Markowitz cost is minimised — a quantum-inspired heuristic.
    # K Ry rotations: P(|1⟩ on qubit i) = sin²(θᵢ/2) ∝ weight of asset i
    theta_vec = ParameterVector('θ', K)
    wt_qc     = QuantumCircuit(K)
    for i in range(K):
        wt_qc.ry(theta_vec[i], i)
    for i in range(K - 1):          # entangle adjacent qubits
        wt_qc.cx(i, i + 1)
    wt_qc.measure_all()

    def circuit_to_weights(theta_vals):
        """Run Ry circuit → marginal P(|1⟩) per qubit → normalised weights."""
        bound = wt_qc.assign_parameters(dict(zip(theta_vec, theta_vals)))
        job   = sampler.run([bound])
        dist  = job.result().quasi_dists[0]
        p = np.zeros(K)
        for state_int, prob in dist.items():
            bits = format(state_int, f'0{K}b')[::-1]   # little-endian
            for qi, b in enumerate(bits):
                if b == '1':
                    p[qi] += max(prob, 0.0)
        total = p.sum()
        return p / total if total > 1e-10 else np.ones(K) / K

    def weight_cost(theta_vals):
        """Markowitz objective using DRO worst-case parameters."""
        w   = circuit_to_weights(theta_vals)
        ret = float(w @ mu_sel_dy)
        var = float(w @ cov_sel_dy @ w)
        return LAMBDA_Q * var - (1 - LAMBDA_Q) * ret

    # Initial angles: π/4 → sin²(π/8)≈0.15 per qubit (roughly equal weights)
    theta0      = np.ones(K) * (np.pi / 4)
    bounds_wt   = [(0.01, np.pi - 0.01)] * K

    # Phase 1: Dual Annealing (global)
    print("  Phase 1 · Dual Annealing ...", end='', flush=True)
    t0   = time.perf_counter()
    res_da = dual_annealing(weight_cost, bounds_wt, x0=theta0,
                            maxiter=200, seed=42,
                            minimizer_kwargs={'method': 'COBYLA'})
    t_da = time.perf_counter() - t0
    print(f" cost={res_da.fun:.5f}  nfev={res_da.nfev}  ({t_da:.1f}s)")

    # Phase 2: COBYLA (local refinement)
    print("  Phase 2 · COBYLA     ...", end='', flush=True)
    t0   = time.perf_counter()
    res_co = sp_min(weight_cost, res_da.x, method='COBYLA',
                    options={'maxiter': 300, 'rhobeg': 0.05})
    t_co = time.perf_counter() - t0
    print(f" cost={res_co.fun:.5f}  nfev={res_co.nfev}  ({t_co:.1f}s)")
    t_s2 = t_da + t_co

    best_theta  = res_co.x if res_co.fun <= res_da.fun else res_da.x
    w_qpw       = circuit_to_weights(best_theta)

    print(f"\n  VQE weight allocation:")
    for name, wi in zip(sel_names, w_qpw):
        bar = '█' * int(wi * 40)
        print(f"    {name:<8} {wi*100:5.1f}%  {bar}")

    # ══════════════════════════════════════════════════════════════
    # STAGE 3 — DRO SHARPE IMPROVEMENT  (DY-DRO SOCP on K assets)
    # ══════════════════════════════════════════════════════════════
    print(f"\nSTAGE 3: DY-DRO Sharpe improvement (SOCP on {K} selected assets)")
    print("-" * 45)

    # Sharpe before DRO (VQE weights, historical parameters)
    r_pre  = float(w_qpw @ mu_sel_std)
    v_pre  = float(w_qpw @ cov_sel_std @ w_qpw)
    sh_pre = (r_pre - RISK_FREE) / np.sqrt(v_pre)

    # DY-DRO SOCP restricted to selected K assets
    Sh_k   = np.linalg.cholesky(cov_sel_dy + 1e-8 * np.eye(K))
    w_dro  = cp.Variable(K, nonneg=True)
    t_cone = cp.Variable(nonneg=True)
    dro_k  = cp.Problem(
        cp.Minimize(LAMBDA_Q * GAMMA_2 * cp.quad_form(w_dro, cov_sel_dy)
                    - (1 - LAMBDA_Q) * mu_sel_dy @ w_dro
                    + (1 - LAMBDA_Q) * np.sqrt(GAMMA_1) * t_cone),
        [cp.sum(w_dro) == 1,
         cp.norm(Sh_k @ w_dro, 2) <= t_cone,
         w_dro <= 1]
    )
    t0 = time.perf_counter()
    dro_k.solve(solver=cp.CLARABEL, verbose=False)
    t_s3 = time.perf_counter() - t0
    w_dro_refined = w_dro.value

    # Sharpe after DRO
    r_post  = float(w_dro_refined @ mu_sel_std)
    v_post  = float(w_dro_refined @ cov_sel_std @ w_dro_refined)
    sh_post = (r_post - RISK_FREE) / np.sqrt(v_post)

    print(f"  Sharpe before DRO (VQE weights)  : {sh_pre:.4f}")
    print(f"  Sharpe after  DRO (SOCP refined) : {sh_post:.4f}  "
          f"({'▲ +' if sh_post>sh_pre else '▼ '}{abs(sh_post-sh_pre):.4f})")
    print(f"\n  DRO-refined weight allocation:")
    for name, wi_pre, wi_post in zip(sel_names, w_qpw, w_dro_refined):
        shift = wi_post - wi_pre
        arrow = f'+{shift*100:.1f}%' if shift >= 0 else f'{shift*100:.1f}%'
        print(f"    {name:<8}  VQE={wi_pre*100:5.1f}%  →  DRO={wi_post*100:5.1f}%  ({arrow})")

    # Final portfolio stats (use DRO-refined weights)
    stats_m8 = {
        'return': r_post,
        'risk'  : np.sqrt(v_post),
        'sharpe': sh_post,
        'tickers': sel_names
    }
    t_m8_total = t_s1 + t_s2 + t_s3

    print(f"\n{'='*55}")
    print(f"  METHOD 8 FINAL RESULT")
    print(f"{'='*55}")
    print(f"  Return  : {r_post*100:.2f}%")
    print(f"  Risk    : {np.sqrt(v_post)*100:.2f}%")
    print(f"  Sharpe  : {sh_post:.4f}")
    print(f"  Time    : {t_s1:.1f}s (S1) + {t_s2:.1f}s (S2) + {t_s3:.2f}s (S3) "
          f"= {t_m8_total:.1f}s total")

    # store results
    for _k in list(locals().keys()):
        if _k.startswith(("stats_m","t_m","t_ws","t_m8")):
            result_d[_k] = locals()[_k]
    for _k in list(locals().keys()):
        if _k.startswith("res_m"):
            try: result_d[_k.replace("res_","ev_")] = float(locals()[_k].eigenvalue)
            except: pass


# =================================================================
# PARALLEL LAUNCHER — 8 methods × 7x H100 80 GB
# =================================================================
if __name__ == "__main__":
    set_start_method("spawn", force=True)   # required for CUDA multiprocessing

    mgr      = Manager()
    result_d = mgr.dict()

    JOBS = [
        (run_method_1, 0),
        (run_method_2, 1),
        (run_method_3, 2),
        (run_method_4, 3),
        (run_method_5, 4),
        (run_method_6, 5),
        (run_method_7, 6),
        (run_method_8, 0),
    ]

    print("\n" + "="*65)
    print(f"  Launching 8 methods in parallel — 7x H100 80 GB")
    print("="*65 + "\n")

    procs = [Process(target=fn, args=(result_d,), name=fn.__name__)
             for fn,_ in JOBS]
    t_wall = time.perf_counter()
    for p in procs: p.start()
    for p in procs: p.join()
    t_wall = time.perf_counter() - t_wall
    print(f"\n  All methods finished — wall clock: {t_wall:.1f}s")

    # unpack results into local namespace for scorecard
    _lcl = locals()
    for _k,_v in result_d.items(): _lcl[_k] = _v


    # =================================================================
    # SCORECARD
    # =================================================================
    # Classical baseline: top-BUDGET_Q by Markowitz weight (equal weight)
    cl_idx    = np.argsort(w_opt)[-BUDGET_Q:][::-1]
    stats_cl  = q_stats(cl_idx)

    methods = {
        'Classical (baseline)':        (stats_cl,   None,  None),
        '1. VQE':                      (stats_m1,   t_m1,  res_m1.eigenvalue),
        '2. QAOA':                     (stats_m2,   t_m2,  res_m2.eigenvalue),
        '3. VQE + QAOA':               (stats_m3,   t_m3,  res_m3.eigenvalue),
        '4. VQE + QAOA + DRO-CVaR':    (stats_m4,   t_m4,  res_m4.eigenvalue),
        '5. + CMA-ES':                 (stats_m5,   t_m5,  res_m5.eigenvalue),
        '6. QAOA+SPA+VQE+DY-DRO':      (stats_m6,   t_ws+t_m6, res_m6.eigenvalue),
        '7. QAOA+SPSA+VQE+DY-DRO':     (stats_m7,   t_ws7+t_m7, res_m7.eigenvalue),
        '8. 3-Stage: QAOA+VQE+DRO':    (stats_m8,   t_m8_total, None),
    }

    print('='*95)
    print('  QUANTUM vs CLASSICAL — 8-METHOD SCORECARD')
    print('='*95)
    print(f"  {'Method':<32} {'Return%':>9} {'Risk%':>8} {'Sharpe':>8} {'ΔSharpe':>9} {'Time(s)':>8} {'Energy':>10}")
    print('-'*95)
    for name, (st, t, ev) in methods.items():
        ds  = st['sharpe'] - stats_cl['sharpe']
        flag = ('▲' if ds > 0 else ('─' if ds == 0 else '▼')) if t is not None else '─'
        t_s  = f"{t:>7.1f}" if t is not None else '      —'
        ev_s = f"{ev:>10.4f}" if ev is not None else '         —'
        print(f"  {name:<32} {st['return']*100:>8.2f}% {st['risk']*100:>7.2f}% "
              f"{st['sharpe']:>8.4f} {flag}{abs(ds):>8.4f} {t_s} {ev_s}")
    print('='*95)

    best_name = max([(n, s) for n, (s, t, _) in methods.items() if t is not None],
                    key=lambda x: x[1]['sharpe'])
    print(f"\n  Best quantum method : {best_name[0]}  (Sharpe={best_name[1]['sharpe']:.4f})")
    print(f"  Classical baseline  : Sharpe={stats_cl['sharpe']:.4f}")

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    names  = list(methods.keys())
    sharpe = [s['sharpe'] for s, _, _ in methods.values()]
    rets   = [s['return']*100 for s, _, _ in methods.values()]
    risks  = [s['risk']*100 for s, _, _ in methods.values()]
    colors = ['#95a5a6'] + ['#3498db', '#2ecc71', '#9b59b6', '#e67e22', '#e74c3c', '#1abc9c', '#f39c12', '#8e44ad']

    for ax, vals, title, ylabel in zip(
            axes,
            [sharpe, rets, risks],
            ['Sharpe Ratio', 'Annual Return %', 'Annual Risk %'],
            ['Sharpe', 'Return (%)', 'Risk (%)']):
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(' + ', '\n+') for n in names], fontsize=7, rotation=30, ha='right')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005*max(vals),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7)

    plt.suptitle('Quantum Portfolio Methods — 5-way Comparison', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'quantum_5method_comparison.png'), dpi=130, bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, f'plot_{int(time.time())}.png'), dpi=130, bbox_inches='tight'); plt.close()
