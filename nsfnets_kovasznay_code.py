"""VP-NSFnet for steady 2D Kovasznay flow (PyTorch).

Reproduction of the velocity-pressure (VP) formulation from
  Jin, Cai, Li, Karniadakis, "NSFnets: Physics-informed neural networks for
  the incompressible Navier-Stokes equations", JCP 2021 (arXiv 2003.06496),
  Sec. 3.1, and the authors' code VP_NFSnets1.py.

Usage
-----
  python nsfnets_kovasznay.py checks     # sanity checks for stages 2-4 + sampling figures
  python nsfnets_kovasznay.py trial      # short training run (2k Adam + 500 L-BFGS)
  python nsfnets_kovasznay.py baseline   # full paper schedule (~1 h on the A5000)
  python nsfnets_kovasznay.py validate --ckpt checkpoints/baseline_final.pt
  python nsfnets_kovasznay.py stage7 --workers 3   # all Stage 7 runs (restartable)
  python nsfnets_kovasznay.py run --exp A_7x50     # one Stage 7 run
  python nsfnets_kovasznay.py summary    # Stage 7 tables -> stage7_summary.md
  python nsfnets_kovasznay.py figures7   # Stage 7 figures fig5, 5b, 5c, 6-8 -> figures/paper/
  python nsfnets_kovasznay.py heat_ref   # heat extension: FD reference T for each Pe
  python nsfnets_kovasznay.py heat_trial --pe 28   # short heat run (2k Adam + 500 L-BFGS)
  python nsfnets_kovasznay.py heat --pe 28         # full heat run (paper Adam + converged L-BFGS)
  python nsfnets_kovasznay.py heat_analysis       # H5: eps_T vs Pe, wall Nu, energy balance, fig9-11

Stage 1 summary: equations (nondimensional, steady, 2D)
-------------------------------------------------------
  continuity:   u_x + v_y = 0
  x-momentum:   u u_x + v u_y + p_x - (1/Re)(u_xx + u_yy) = 0
  y-momentum:   u v_x + v v_y + p_y - (1/Re)(v_xx + v_yy) = 0
The network N(x, y) -> (u, v, p) is trained so that these residuals vanish at
interior points and u, v match the exact solution on the boundary:
  L = L_e + alpha * L_b         (paper Eq. 2a; no initial-condition term, flow is steady)

Each "# %%" marker starts a section (also runnable as a cell in VS Code).
"""
# %% Imports and global settings
import argparse
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")                    # write figures to files; no display needed
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIG_DIR = "figures"
CKPT_DIR = "checkpoints"


# %% Stage 2a: problem constants and exact solution
# Re = 40 and the domain follow the paper (Sec. 3.1) and the authors' code.
RE = 40.0
LAM = RE / 2 - math.sqrt(RE**2 / 4 + 4 * math.pi**2)     # lambda ~ -0.9637
X_MIN, X_MAX = -0.5, 1.0
Y_MIN, Y_MAX = -0.5, 1.5


def exact_solution(x, y):
    """Analytical Kovasznay fields. x, y: (N, 1) -> u, v, p: (N, 1)."""
    exp_lx = torch.exp(LAM * x)                          # shared factor e^{lambda x}
    u = 1.0 - exp_lx * torch.cos(2 * math.pi * y)
    v = LAM / (2 * math.pi) * exp_lx * torch.sin(2 * math.pi * y)
    p = 0.5 * (1.0 - torch.exp(2 * LAM * x))
    return u, v, p


# %% Stage 2b: training and validation points
N_INTERIOR = 2601    # paper: "2601 points are randomly selected inside the domain"
N_PER_EDGE = 101     # paper: 101 per edge -> N_b = 400 (each corner once)
N_VAL = 201          # our choice: 201 x 201 validation grid


def interior_points(n=N_INTERIOR, seed=1234):
    """A. Uniform random collocation points (no labels). Returns x, y: (n, 1)."""
    gen = torch.Generator().manual_seed(seed)    # own generator: independent of model init
    x = X_MIN + (X_MAX - X_MIN) * torch.rand(n, 1, generator=gen)
    y = Y_MIN + (Y_MAX - Y_MIN) * torch.rand(n, 1, generator=gen)
    return x, y


def boundary_points(n_per_edge=N_PER_EDGE):
    """B. Evenly spaced edge points with exact u, v (no pressure).

    Each edge drops one end point so every corner appears exactly once:
    4 * (n_per_edge - 1) = 400 points, same construction as the authors' code.
    """
    xs = torch.linspace(X_MIN, X_MAX, n_per_edge)
    ys = torch.linspace(Y_MIN, Y_MAX, n_per_edge)
    m = n_per_edge - 1
    #            x coordinates                 y coordinates
    left   = (torch.full((m,), X_MIN),       ys[1:])
    right  = (torch.full((m,), X_MAX),       ys[:-1])
    bottom = (xs[:-1],                       torch.full((m,), Y_MIN))
    top    = (xs[1:],                        torch.full((m,), Y_MAX))
    edges = (left, right, bottom, top)
    x = torch.cat([e[0] for e in edges]).reshape(-1, 1)     # (400, 1)
    y = torch.cat([e[1] for e in edges]).reshape(-1, 1)     # (400, 1)
    u, v, _ = exact_solution(x, y)                          # pressure deliberately dropped
    return x, y, u, v


def validation_grid(n=N_VAL):
    """C. Regular n x n grid with full exact solution, for evaluation only.

    Flattened (n*n, 1) tensors for the network + 2D (n, n) mesh X, Y for plots.
    """
    xs = torch.linspace(X_MIN, X_MAX, n)
    ys = torch.linspace(Y_MIN, Y_MAX, n)
    X, Y = torch.meshgrid(xs, ys, indexing="xy")    # (n, n); rows = y, cols = x
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    u, v, p = exact_solution(x, y)
    return {"x": x, "y": y, "u": u, "v": v, "p": p, "X": X, "Y": Y}


def plot_sampling():
    xi, yi = interior_points()
    xb, yb, _, _ = boundary_points()
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.scatter(xi, yi, s=2, c="tab:blue", label=f"interior collocation ({len(xi)})")
    ax.scatter(xb, yb, s=6, c="tab:red", label=f"boundary, exact u,v ({len(xb)})")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Training points")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "sampling.png"), dpi=150)
    plt.close(fig)

    val = validation_grid()
    n = val["X"].shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, name in zip(axes, ("u", "v", "p")):
        cf = ax.contourf(val["X"], val["Y"], val[name].reshape(n, n), levels=50, cmap="viridis")
        fig.colorbar(cf, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(f"exact {name}")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "exact_fields.png"), dpi=150)
    plt.close(fig)


# %% Stage 3: the VP network (x, y) -> (u, v, p)
class VPNet(nn.Module):
    """Fully connected tanh network; baseline 4 x 50 as in the paper/code."""

    def __init__(self, hidden_layers=4, width=50,
                 lb=(X_MIN, Y_MIN), ub=(X_MAX, Y_MAX), n_out=3):
        super().__init__()
        # Domain bounds: not trained, but move with .to(device) and are saved.
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))

        self.n_out = n_out                         # 3: (u, v, p); heat extension: 4 (+ T)
        sizes = [2] + [width] * hidden_layers + [n_out]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:                 # no activation on the output layer
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Xavier as in the authors' code: truncated normal (2 std),
        std = sqrt(2 / (fan_in + fan_out)); zero biases."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                fan_out, fan_in = m.weight.shape       # PyTorch stores (out, in)
                std = math.sqrt(2.0 / (fan_in + fan_out))
                nn.init.trunc_normal_(m.weight, std=std, a=-2 * std, b=2 * std)
                nn.init.zeros_(m.bias)

    def forward(self, x, y):
        """x, y: physical coordinates (N, 1) -> u, v, p: (N, 1) each."""
        X = torch.cat([x, y], dim=1)                            # (N, 2)
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0     # (N, 2) in [-1, 1]
        out = self.net(H)                                        # (N, n_out)
        return tuple(out[:, i:i + 1] for i in range(self.n_out))  # i:i+1 keeps (N, 1)


# %% Stage 4: physics-informed loss
def grad(f, x):
    """Pointwise df/dx for f, x of shape (N, 1).

    create_graph=True keeps the derivative in the graph: needed for second
    derivatives and for backpropagating the physics loss to the weights.
    allow_unused=True only matters for the exact p, which has no y-dependence.
    """
    df = torch.autograd.grad(f, x, grad_outputs=torch.ones_like(f),
                             create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(x) if df is None else df


def pde_residuals(field_fn, x, y):
    """Residuals r_c, r_u, r_v (each (N, 1)) of any callable (x, y) -> (u, v, p)."""
    x = x.detach().requires_grad_(True)       # fresh leaves; caller's tensors untouched
    y = y.detach().requires_grad_(True)
    u, v, p = field_fn(x, y)

    u_x, u_y = grad(u, x), grad(u, y)
    v_x, v_y = grad(v, x), grad(v, y)
    p_x, p_y = grad(p, x), grad(p, y)
    u_xx, u_yy = grad(u_x, x), grad(u_y, y)
    v_xx, v_yy = grad(v_x, x), grad(v_y, y)

    r_c = u_x + v_y
    r_u = u * u_x + v * u_y + p_x - (u_xx + u_yy) / RE
    r_v = u * v_x + v * v_y + p_y - (v_xx + v_yy) / RE
    return r_c, r_u, r_v


def total_loss(field_fn, interior, boundary, alpha=100.0):
    """L = L_e + alpha * L_b. Returns (loss, dict of detached components)."""
    r_c, r_u, r_v = pde_residuals(field_fn, *interior)
    L_c, L_u, L_v = r_c.pow(2).mean(), r_u.pow(2).mean(), r_v.pow(2).mean()
    L_e = L_c + L_u + L_v

    xb, yb, ub, vb = boundary
    u_pred, v_pred, _ = field_fn(xb, yb)       # no pressure boundary condition
    L_bu = (u_pred - ub).pow(2).mean()
    L_bv = (v_pred - vb).pow(2).mean()
    L_b = L_bu + L_bv

    loss = L_e + alpha * L_b
    # Detached tensors, not .item(): avoids a GPU->CPU sync on every step.
    parts = {"total": loss, "L_e": L_e, "L_b": L_b, "L_c": L_c, "L_u": L_u,
             "L_v": L_v, "L_bu": L_bu, "L_bv": L_bv}
    return loss, {k: t.detach() for k, t in parts.items()}


# %% Stage 5a: evaluation metric (relative L2 error, pressure mean-shifted)
@torch.no_grad()
def relative_errors(model, val):
    """eps = ||pred - exact||_2 / ||exact||_2 on the validation grid (paper Eq. 9).

    p is known only up to a constant (its output-bias gradient is exactly 0),
    so we shift the predicted p to have the same mean as the exact p, as the
    paper describes. The authors' code skips this shift.
    """
    u, v, p = model(val["x"], val["y"])
    p = p - p.mean() + val["p"].mean()
    rel = lambda a, b: (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()
    return {"eps_u": rel(u, val["u"]), "eps_v": rel(v, val["v"]), "eps_p": rel(p, val["p"])}


# %% Stage 5b: training loops
# Paper Sec. 3.1: Adam 5k @ 1e-3, 5k @ 1e-4, 50k @ 1e-5, 50k @ 1e-6, then L-BFGS-B.
PAPER_ADAM = [(5_000, 1e-3), (5_000, 1e-4), (50_000, 1e-5), (50_000, 1e-6)]


def _record(history, step, phase, parts, errors=None):
    row = {"step": step, "phase": phase, **{k: float(t) for k, t in parts.items()}}
    if errors:
        row.update(errors)
    history.append(row)
    if not math.isfinite(row["total"]):
        raise RuntimeError(f"Loss became {row['total']} at step {step} ({phase})")
    return row


def _print_row(row):
    msg = (f"[{row['phase']:>9}] step {row['step']:7d} | loss {row['total']:.3e} "
           f"| L_e {row['L_e']:.3e} | L_b {row['L_b']:.3e}")
    if "L_T" in row:
        msg += f" | L_T {row['L_T']:.3e} | L_bT {row['L_bT']:.3e}"
    if "eps_u" in row:
        msg += (f" | eps_u {100 * row['eps_u']:.3f}% eps_v {100 * row['eps_v']:.3f}%"
                f" eps_p {100 * row['eps_p']:.3f}%")
    if "eps_T" in row:
        msg += f" eps_T {100 * row['eps_T']:.3f}%"
    print(msg, flush=True)


def save_checkpoint(path, model, history, info):
    torch.save({"model": model.state_dict(), "history": history, "info": info}, path)


def train_adam(model, interior, boundary, val, schedule, alpha, history,
               log_every=100, eval_every=1000, tag=None, loss_fn=None, error_fn=None):
    """Adam with a piecewise-constant learning-rate schedule [(n_iter, lr), ...].

    One optimizer for all phases (moment estimates carry over); only the lr
    changes between phases. A checkpoint is saved at the end of each phase.
    """
    # loss_fn() -> (loss, parts), error_fn() -> dict: default = the flow problem
    loss_fn = loss_fn or (lambda: total_loss(model, interior, boundary, alpha))
    error_fn = error_fn or (lambda: relative_errors(model, val))
    opt = torch.optim.Adam(model.parameters(), lr=schedule[0][1])
    step = 0
    for k, (n_iter, lr) in enumerate(schedule):
        for group in opt.param_groups:
            group["lr"] = lr
        t0 = time.time()
        for _ in range(n_iter):
            opt.zero_grad(set_to_none=True)
            loss, parts = loss_fn()
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                errors = error_fn() if step % eval_every == 0 else None
                row = _record(history, step, f"adam{k + 1}", parts, errors)
                if errors:
                    _print_row(row)
        print(f"  Adam phase {k + 1}: {n_iter} iters at lr={lr:g} in {time.time() - t0:.0f} s")
        if tag:
            save_checkpoint(os.path.join(CKPT_DIR, f"{tag}_adam{k + 1}.pt"), model, history,
                            {"phase": f"adam{k + 1}", "step": step})
    return step


def train_lbfgs(model, interior, boundary, val, alpha, history, step0,
                max_iter=50_000, log_every=100, eval_every=1000,
                plateau=None, chunk=1000, loss_fn=None, error_fn=None):
    """L-BFGS fine-tuning, mirroring the authors' SciPy L-BFGS-B settings where
    PyTorch allows: maxiter = maxfun = max_iter, maxcor (history) = 50, ftol = eps,
    gtol = SciPy default 1e-5. Steps are counted as loss evaluations.

    Loss scaling: L-BFGS minimises loss / loss_0 (same minimiser). PyTorch only
    stores a curvature pair if y.s > 1e-10 (absolute, hard-coded); with raw losses
    ~1e-5 the accepted steps give y.s below that, the history never fills and
    L-BFGS stalls with the weights frozen (happened in the first A_7x100,
    A_10x100, A_10x200 runs). SciPy's test is relative, so it has no such issue.
    gtol is rescaled so it still means max|grad of raw loss| < 1e-5.

    plateau=(window, rel): run in chunks of `chunk` evaluations and stop once the
    loss fell by less than the fraction `rel` over the last `window` evaluations
    ("converged"), or on PyTorch's own tolerances, or at max_iter evaluations.
    """
    loss_fn = loss_fn or (lambda: total_loss(model, interior, boundary, alpha))
    error_fn = error_fn or (lambda: relative_errors(model, val))
    loss0 = loss_fn()[0].item()               # needs autograd (PDE derivatives)
    opt = torch.optim.LBFGS(
        model.parameters(), lr=1.0,
        max_iter=chunk if plateau else max_iter,
        max_eval=chunk if plateau else max_iter,
        history_size=50,
        tolerance_grad=1e-5 / loss0,
        tolerance_change=torch.finfo(torch.float64).eps,
        line_search_fn="strong_wolfe",
    )
    n_eval = 0
    last = {}

    def closure():
        nonlocal n_eval
        opt.zero_grad(set_to_none=True)
        loss, parts = loss_fn()
        (loss / loss0).backward()
        n_eval += 1
        last["loss"] = loss.item()
        if n_eval % log_every == 0:
            errors = error_fn() if n_eval % eval_every == 0 else None
            row = _record(history, step0 + n_eval, "lbfgs", parts, errors)
            if errors:
                _print_row(row)
        return loss / loss0

    t0 = time.time()
    reason = "max evaluations"
    if not plateau:
        opt.step(closure)
    else:
        window, rel = plateau
        chunk_losses = []
        while n_eval < max_iter:
            before = n_eval
            opt.step(closure)
            chunk_losses.append(last["loss"])     # loss at the last evaluated point
            if n_eval - before < chunk:          # stopped on its own tolerances
                reason = "PyTorch tolerance"
                break
            k = window // chunk
            if len(chunk_losses) > k:
                old, new = chunk_losses[-k - 1], chunk_losses[-1]
                if (old - new) / old < rel:
                    reason = f"plateau (<{100 * rel:g}% drop over {window} evals)"
                    break
    state = next(iter(opt.state.values()))
    print(f"  L-BFGS: {state['n_iter']} iterations, {n_eval} loss evaluations "
          f"in {time.time() - t0:.0f} s; stopped: {reason}")
    return step0 + n_eval


def run_experiment(tag, adam_schedule, lbfgs_iters, alpha=100.0, seed=1234,
                   hidden_layers=4, width=50, n_interior=N_INTERIOR,
                   n_per_edge=N_PER_EDGE, resume_from=None, plateau=None):
    """Build a fresh network, train with Adam then L-BFGS, save and report.

    resume_from: path to an Adam checkpoint; if given, its weights and history
    are loaded and the Adam stage is skipped (only L-BFGS runs).
    """
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    interior = tuple(t.to(DEVICE) for t in interior_points(n_interior))
    boundary = tuple(t.to(DEVICE) for t in boundary_points(n_per_edge))
    val = {k: t.to(DEVICE) for k, t in validation_grid().items()}

    torch.manual_seed(seed)                     # controls the weight initialisation
    model = VPNet(hidden_layers, width).to(DEVICE)
    info = {"tag": tag, "adam_schedule": adam_schedule, "lbfgs_iters": lbfgs_iters,
            "alpha": alpha, "seed": seed, "hidden_layers": hidden_layers, "width": width,
            "n_interior": n_interior, "n_boundary": 4 * (n_per_edge - 1),
            "resume_from": resume_from, "plateau": plateau}
    print(f"=== {tag}: {info} on {DEVICE} ===")

    t0 = time.time()
    if resume_from:
        ckpt = torch.load(resume_from, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
        history = ckpt["history"]
        step = ckpt["info"]["step"]
        print(f"Resumed from {resume_from} at step {step}; skipping Adam")
    else:
        history = []
        step = train_adam(model, interior, boundary, val, adam_schedule, alpha, history, tag=tag)

    if lbfgs_iters > 0:
        # L-BFGS in float64. In float32 its strong-Wolfe line search fails near
        # the minimum (returns step size 0) and PyTorch stops after a handful of
        # iterations with the gradient still ~400x above tolerance. The authors'
        # SciPy L-BFGS-B also does its internal arithmetic in float64.
        model.double()
        interior = tuple(t.double() for t in interior)
        boundary = tuple(t.double() for t in boundary)
        val = {k: t.double() for k, t in val.items()}
        step = train_lbfgs(model, interior, boundary, val, alpha, history, step,
                           max_iter=lbfgs_iters, plateau=plateau)
    info["train_seconds"] = time.time() - t0

    # Final state: one more loss evaluation + validation errors, logged as the last row
    _, parts = total_loss(model, interior, boundary, alpha)
    errors = relative_errors(model, val)
    _record(history, step, history[-1]["phase"], parts, errors)
    info.update(errors)
    save_checkpoint(os.path.join(CKPT_DIR, f"{tag}_final.pt"), model, history, info)
    plot_history(history, os.path.join(FIG_DIR, f"{tag}_history.png"), title=tag)
    print(f"Final: eps_u {100 * errors['eps_u']:.3f}%  eps_v {100 * errors['eps_v']:.3f}%  "
          f"eps_p {100 * errors['eps_p']:.3f}%   ({info['train_seconds'] / 60:.1f} min)")
    return model, history, info


def plot_history(history, path, title=""):
    """Loss components (left) and validation errors (right) vs. step."""
    steps = [r["step"] for r in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in (("total", "total"), ("L_e", "L_e (PDE)"), ("L_b", "L_b (boundary)")):
        ax1.semilogy(steps, [r[key] for r in history], label=label, lw=1)
    ev = [r for r in history if "eps_u" in r]
    for key in ("eps_u", "eps_v", "eps_p"):
        ax2.semilogy([r["step"] for r in ev], [r[key] for r in ev], "o-", ms=2, lw=1, label=key)
    # mark phase changes (at the last step of the finished phase)
    for i in range(1, len(history)):
        if history[i]["phase"] != history[i - 1]["phase"]:
            for ax in (ax1, ax2):
                ax.axvline(history[i - 1]["step"], color="gray", ls=":", lw=0.8)
    ax1.set_title(f"{title}: loss")
    ax2.set_title(f"{title}: relative L2 error (validation)")
    for ax in (ax1, ax2):
        ax.set_xlabel("step (Adam iteration / L-BFGS evaluation)")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# %% Stage 6: validation and visualisation of a trained model
def load_model(path):
    """Rebuild a VPNet from a checkpoint (architecture read from its info).
    Returned in float64: evaluation is cheap, so use full precision."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    info = ckpt["info"]
    model = VPNet(info.get("hidden_layers", 4), info.get("width", 50)).to(DEVICE).double()
    model.load_state_dict(ckpt["model"])     # copies (and casts) the saved weights
    model.eval()
    return model, ckpt["history"], info


def plot_fields(model, val, path):
    """Rows u, v, p; columns exact | predicted | |error|. Pressure mean-shifted."""
    n = val["X"].shape[0]
    with torch.no_grad():
        u, v, p = model(val["x"], val["y"])
    p = p - p.mean() + val["p"].mean()
    X, Y = val["X"].cpu(), val["Y"].cpu()
    fig, axes = plt.subplots(3, 3, figsize=(12, 13))
    for row, (name, pred) in enumerate((("u", u), ("v", v), ("p", p))):
        exact = val[name].reshape(n, n).cpu()
        pred = pred.reshape(n, n).cpu()
        err = (pred - exact).abs()
        vmin, vmax = exact.min().item(), exact.max().item()     # same colour scale for both
        for col, (field, title, cmap, lims) in enumerate((
                (exact, f"exact {name}", "viridis", (vmin, vmax)),
                (pred, f"predicted {name}", "viridis", (vmin, vmax)),
                (err, f"|error| {name}  (max {err.max():.1e})", "inferno", (None, None)))):
            ax = axes[row, col]
            levels = 50 if lims[0] is None else torch.linspace(vmin, vmax, 51).numpy()
            cf = ax.contourf(X, Y, field, levels=levels, cmap=cmap, extend="both")
            fig.colorbar(cf, ax=ax, format="%.1e" if col == 2 else None)
            ax.set_aspect("equal")
            ax.set_title(title)
            if row == 2:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_residual_maps(model, val, path):
    """|r_c|, |r_u|, |r_v| over the whole validation grid (mostly unseen points)."""
    n = val["X"].shape[0]
    residuals = pde_residuals(model, val["x"], val["y"])
    X, Y = val["X"].cpu(), val["Y"].cpu()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    for ax, name, r in zip(axes, ("r_c (continuity)", "r_u (x-momentum)", "r_v (y-momentum)"), residuals):
        field = r.detach().abs().reshape(n, n).cpu()
        cf = ax.contourf(X, Y, field, levels=50, cmap="inferno")
        fig.colorbar(cf, ax=ax, format="%.1e")
        ax.set_aspect("equal")
        ax.set_title(f"|{name}|")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def residual_stats(model, x, y):
    """RMS and max |residual| for each equation at the given points."""
    out = {}
    for name, r in zip(("r_c", "r_u", "r_v"), pde_residuals(model, x, y)):
        r = r.detach()
        out[name] = (r.pow(2).mean().sqrt().item(), r.abs().max().item())
    return out


def run_validation(ckpt_path):
    os.makedirs(FIG_DIR, exist_ok=True)
    model, history, info = load_model(ckpt_path)
    tag = info.get("tag", "model")
    dt = torch.float64
    val = {k: t.to(DEVICE, dt) for k, t in validation_grid().items()}
    xi, yi = (t.to(DEVICE, dt) for t in interior_points())
    print(f"=== Validation of {ckpt_path} (tag '{tag}') ===")

    # 1. Relative L2 errors: whole grid, and interior nodes only (no boundary nodes)
    full = relative_errors(model, val)
    on_edge = ((val["x"] == X_MIN) | (val["x"] == X_MAX)
               | (val["y"] == Y_MIN) | (val["y"] == Y_MAX)).squeeze(1)
    inner = {k: (t[~on_edge] if t.shape[0] == on_edge.shape[0] else t) for k, t in val.items()}
    interior_only = relative_errors(model, inner)
    print(f"\n1. Relative L2 error (%)       eps_u     eps_v     eps_p")
    print(f"   whole grid ({on_edge.numel()} pts)   "
          + "  ".join(f"{100 * full[k]:8.4f}" for k in ("eps_u", "eps_v", "eps_p")))
    print(f"   interior only ({int((~on_edge).sum())} pts)"
          + "  ".join(f"{100 * interior_only[k]:8.4f}" for k in ("eps_u", "eps_v", "eps_p")))
    print("   paper, 4x50 (best of 10)        0.076     0.412     0.516")

    # 2. Where is the error largest?
    with torch.no_grad():
        u, v, p = model(val["x"], val["y"])
    p = p - p.mean() + val["p"].mean()
    print("\n2. Maximum absolute error and its location")
    for name, pred in (("u", u), ("v", v), ("p", p)):
        err = (pred - val[name]).abs().squeeze(1)
        i = int(err.argmax())
        scale = val[name].abs().max().item()
        print(f"   {name}: max |error| {err[i].item():.2e} ({100 * err[i].item() / scale:.3f}% of max|{name}|)"
              f" at (x, y) = ({val['x'][i].item():+.3f}, {val['y'][i].item():+.3f})")

    # 3. Overfitting check: residuals at training points vs. unseen grid points
    seen = residual_stats(model, xi, yi)
    unseen = residual_stats(model, inner["x"], inner["y"])
    print("\n3. PDE residuals (RMS / max |r|): training points vs. unseen grid points")
    for name in ("r_c", "r_u", "r_v"):
        print(f"   {name}:  training {seen[name][0]:.2e} / {seen[name][1]:.2e}    "
              f"unseen {unseen[name][0]:.2e} / {unseen[name][1]:.2e}    "
              f"RMS ratio unseen/train = {unseen[name][0] / seen[name][0]:.2f}")

    # 4. Figures
    plot_fields(model, val, os.path.join(FIG_DIR, f"{tag}_fields.png"))
    plot_residual_maps(model, val, os.path.join(FIG_DIR, f"{tag}_residuals.png"))
    print(f"\n4. Saved {FIG_DIR}/{tag}_fields.png and {FIG_DIR}/{tag}_residuals.png")


# %% Publication figures (paper / slides): vector PDF + 300 dpi PNG in figures/paper/
PAPER_FIG_DIR = os.path.join(FIG_DIR, "paper")
PAPER_TABLE1_4X50 = {"eps_u": 0.076, "eps_v": 0.412, "eps_p": 0.516}    # %, best of 10 runs
FULL_W = 7.0                                   # inches: full text width of an A4/letter paper
SERIES = {"u": "#1f5fa8", "v": "#d1495b", "p": "#2a9d8f"}               # one colour per variable


def _paper_style():
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm", "font.size": 9,
        "axes.titlesize": 9.5, "axes.labelsize": 9.5, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7,
        "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,                    # embed TrueType: editable text in the PDF
    })


def _save(fig, name):
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    for ext, dpi in (("pdf", None), ("png", 300)):
        fig.savefig(os.path.join(PAPER_FIG_DIR, f"{name}.{ext}"), dpi=dpi)
    plt.close(fig)
    print(f"  saved {PAPER_FIG_DIR}/{name}.pdf / .png")


def _field_axes(ax, xlabel=True, ylabel=True):
    ax.set_aspect("equal")
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_yticks([-0.5, 0, 0.5, 1.0, 1.5])
    ax.set_rasterization_zorder(0)            # contour fills rasterised, text/axes vector
    if xlabel:
        ax.set_xlabel("$x$")
    else:
        ax.set_xticklabels([])
    if ylabel:
        ax.set_ylabel("$y$")
    else:
        ax.set_yticklabels([])


def _panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom",
            fontweight="bold")


def fig_setup(val):
    """(a) training points; (b) exact speed with streamlines."""
    import numpy as np
    xi, yi = interior_points()
    xb, yb, _, _ = boundary_points()
    n = val["X"].shape[0]
    X, Y = val["X"].cpu().numpy(), val["Y"].cpu().numpy()
    U = val["u"].reshape(n, n).cpu().numpy()
    V = val["v"].reshape(n, n).cpu().numpy()

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W * 0.8, 3.6), constrained_layout=True)
    a.scatter(xi, yi, s=1.2, c="#4a6fa5", lw=0, label=f"interior collocation ($N_e$ = {len(xi)})")
    a.scatter(xb, yb, s=3, c="#d1495b", lw=0, label=f"boundary, exact $u,v$ ($N_b$ = {len(xb)})")
    a.set_xlim(X_MIN - 0.05, X_MAX + 0.05)
    a.set_ylim(Y_MIN - 0.05, Y_MAX + 0.05)
    _field_axes(a)
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, handletextpad=0.2,
             markerscale=4)
    _panel(a, "a")

    cf = b.contourf(X, Y, np.hypot(U, V), levels=40, cmap="viridis", zorder=-1)
    b.streamplot(X, Y, U, V, color="white", linewidth=0.6, density=1.1, arrowsize=0.6)
    b.set_xlim(X_MIN, X_MAX)
    b.set_ylim(Y_MIN, Y_MAX)
    _field_axes(b, ylabel=False)
    fig.colorbar(cf, ax=b, shrink=0.85, label=r"$|\mathbf{u}|$ (exact)", ticks=[0, 0.5, 1, 1.5, 2, 2.5])
    _panel(b, "b")
    _save(fig, "fig1_setup")


def fig_fields(model, val):
    """Rows u, v, p; columns exact | PINN | |error|. Shared colour bar for exact and PINN."""
    import numpy as np
    n = val["X"].shape[0]
    with torch.no_grad():
        u, v, p = model(val["x"], val["y"])
    p = p - p.mean() + val["p"].mean()
    X, Y = val["X"].cpu().numpy(), val["Y"].cpu().numpy()

    fig = plt.figure(figsize=(FULL_W, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(3, 3)
    letters = iter("abcdefghi")
    for row, (name, pred) in enumerate((("u", u), ("v", v), ("p", p))):
        exact = val[name].reshape(n, n).cpu().numpy()
        pred = pred.reshape(n, n).cpu().numpy()
        err = np.abs(pred - exact)
        levels = np.linspace(exact.min(), exact.max(), 41)
        last = row == 2
        axes = [fig.add_subplot(grid[row, c]) for c in range(3)]
        for c, (field, title) in enumerate(((exact, f"exact ${name}$"), (pred, f"PINN ${name}$"))):
            cf = axes[c].contourf(X, Y, field, levels=levels, cmap="viridis", extend="both",
                                  zorder=-1)
            axes[c].set_title(title)
            _field_axes(axes[c], xlabel=last, ylabel=c == 0)
            _panel(axes[c], next(letters))
        fig.colorbar(cf, ax=axes[:2], shrink=0.9, aspect=25,
                     ticks=np.linspace(exact.min(), exact.max(), 5), format="%.2f")
        ce = axes[2].contourf(X, Y, err, levels=40, cmap="magma", zorder=-1)
        axes[2].set_title(f"$|{name}_{{\\mathrm{{PINN}}}} - {name}_{{\\mathrm{{exact}}}}|$")
        _field_axes(axes[2], xlabel=last, ylabel=False)
        _panel(axes[2], next(letters))
        cb = fig.colorbar(ce, ax=axes[2], shrink=0.9, aspect=25)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position("left")
        cb.update_ticks()
    _save(fig, "fig2_fields")


def fig_history(history):
    """Loss components and validation errors vs. iteration, phases shaded."""
    steps = [r["step"] / 1e3 for r in history]
    # phase spans [start, end] in thousands of steps
    spans, start = [], 0.0
    for i in range(1, len(history) + 1):
        if i == len(history) or history[i]["phase"] != history[i - 1]["phase"]:
            spans.append((history[i - 1]["phase"], start, history[i - 1]["step"] / 1e3))
            start = history[i - 1]["step"] / 1e3
    lr = {"adam1": "$10^{-3}$", "adam2": "$10^{-4}$", "adam3": "$10^{-5}$", "adam4": "$10^{-6}$"}

    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.9), constrained_layout=True)
    for ax in (a, b):
        for k, (phase, s, e) in enumerate(spans):
            ax.axvspan(s, e, color="#9aa5b1" if phase == "lbfgs" else "#cfd6dd",
                       alpha=0.18 if k % 2 == 0 else 0.35, lw=0)
            label = "L-BFGS" if phase == "lbfgs" else f"Adam\nlr = {lr.get(phase, phase)}"
            if e - s > 15:
                ax.text((s + e) / 2, 0.97, label, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=7, color="#44505c")
        # first two Adam phases (0-10k, lr 1e-3 / 1e-4) are too narrow to label: see caption
        ax.set_xlim(0, steps[-1])
        ax.set_xlabel(r"iteration ($\times 10^3$)")
        ax.grid(alpha=0.25, which="major", lw=0.5)

    for key, label, col in (("total", r"$\mathcal{L}=\mathcal{L}_e+\alpha\,\mathcal{L}_b$", "black"),
                            ("L_e", r"$\mathcal{L}_e$ (PDE residual)", "#e07a1f"),
                            ("L_b", r"$\mathcal{L}_b$ (boundary)", "#2a9d8f")):
        a.semilogy(steps, [r[key] for r in history], color=col, lw=0.9, label=label)
    a.set_ylabel("loss")
    a.legend(loc="lower left", bbox_to_anchor=(0.1, 0.0))
    _panel(a, "a")

    ev = [r for r in history if "eps_u" in r]
    es = [r["step"] / 1e3 for r in ev]
    for name in ("u", "v", "p"):
        k = f"eps_{name}"
        b.semilogy(es, [100 * r[k] for r in ev], color=SERIES[name], lw=1.1,
                   label=f"$\\varepsilon_{name}$")
        b.axhline(PAPER_TABLE1_4X50[k], color=SERIES[name], ls="--", lw=0.8)
    b.plot([], [], color="gray", ls="--", lw=0.8, label="paper, Table 1")
    b.set_ylabel(r"relative $L_2$ error (%)")
    b.legend(loc="lower left", bbox_to_anchor=(0.1, 0.0), ncol=2)
    _panel(b, "b")
    _save(fig, "fig3_history")


def fig_residuals(model, val):
    """|r_c|, |r_u|, |r_v| on the validation grid, log colour scale."""
    import numpy as np
    from matplotlib.colors import LogNorm
    n = val["X"].shape[0]
    X, Y = val["X"].cpu().numpy(), val["Y"].cpu().numpy()
    res = [r.detach().abs().reshape(n, n).cpu().numpy() for r in pde_residuals(model, val["x"], val["y"])]
    hi = max(r.max() for r in res)
    lo = hi * 1e-4
    levels = np.logspace(np.log10(lo), np.log10(hi), 41)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.1), constrained_layout=True)
    names = (r"$|r_c|$ (continuity)", r"$|r_u|$ ($x$-momentum)", r"$|r_v|$ ($y$-momentum)")
    for k, (ax, title, r) in enumerate(zip(axes, names, res)):
        cf = ax.contourf(X, Y, np.clip(r, lo, hi), levels=levels, norm=LogNorm(lo, hi),
                         cmap="magma", zorder=-1)
        ax.set_title(title)
        _field_axes(ax, ylabel=k == 0)
        _panel(ax, "abc"[k])
    fig.colorbar(cf, ax=axes, shrink=0.85, aspect=25, label="absolute residual",
                 ticks=[10.0**e for e in range(int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi))) + 1)])
    _save(fig, "fig4_residuals")


def make_paper_figures(ckpt_path):
    _paper_style()
    model, history, info = load_model(ckpt_path)
    val = {k: t.to(DEVICE, torch.float64) for k, t in validation_grid().items()}
    print(f"Paper figures from {ckpt_path} -> {PAPER_FIG_DIR}/")
    fig_setup(val)
    fig_fields(model, val)
    fig_history(history)
    fig_residuals(model, val)


# %% Sanity checks for stages 2-4
def run_checks():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Re = {RE}, lambda = {LAM:.6f}, device = {DEVICE}")

    # Stage 2: points
    xi, yi = interior_points()
    xb, yb, ub, vb = boundary_points()
    val = validation_grid()
    print(f"\n[Stage 2] interior {tuple(xi.shape)}, boundary {tuple(xb.shape)}, "
          f"validation {tuple(val['x'].shape)} (mesh {tuple(val['X'].shape)})")
    inside = bool((xi >= X_MIN).all() and (xi <= X_MAX).all()
                  and (yi >= Y_MIN).all() and (yi <= Y_MAX).all())
    n_unique = torch.unique(torch.cat([xb, yb], 1), dim=0).shape[0]
    corners = [int(((xb == cx) & (yb == cy)).sum())
               for cx in (X_MIN, X_MAX) for cy in (Y_MIN, Y_MAX)]
    print(f"  interior inside domain: {inside}; unique boundary points: {n_unique}; "
          f"corner counts: {corners}")
    plot_sampling()
    print(f"  saved {FIG_DIR}/sampling.png, {FIG_DIR}/exact_fields.png")

    # Stage 3: network
    torch.manual_seed(1234)
    model = VPNet()
    u, v, p = model(xi, yi)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"\n[Stage 3] parameters: {n_params}; output shapes {tuple(u.shape)} x 3")
    m64 = VPNet().double()
    xs = torch.tensor([[0.3]], dtype=torch.float64, requires_grad=True)
    ys = torch.tensor([[0.7]], dtype=torch.float64)
    u0 = m64(xs, ys)[0]
    u_x = torch.autograd.grad(u0, xs, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, xs)[0]
    h = 1e-4
    with torch.no_grad():
        up, um = m64(xs + h, ys)[0], m64(xs - h, ys)[0]
    print(f"  u_x : autograd {u_x.item():+.8f}  finite-diff {((up - um) / (2 * h)).item():+.8f}")
    print(f"  u_xx: autograd {u_xx.item():+.8f}  finite-diff {((up - 2 * u0 + um) / h**2).item():+.8f}")

    # Stage 4: loss
    interior, boundary = (xi, yi), (xb, yb, ub, vb)
    _, parts = total_loss(exact_solution, interior, boundary)
    print(f"\n[Stage 4] exact solution: total loss {float(parts['total']):.2e} (round-off)")
    loss, parts = total_loss(model, interior, boundary)
    print(f"  untrained net: L_e {float(parts['L_e']):.3e}, alpha*L_b {100 * float(parts['L_b']):.3e}")
    loss.backward()
    print(f"  output-bias gradients (u, v, p): "
          f"{[round(g, 4) for g in model.net[-1].bias.grad.tolist()]}  <- p entry exactly 0")


# %% Stage 7: parametric study (one factor at a time around the baseline)
# Every run below differs from the reference in ONE setting, except
# D_user_original, which is the user's original plan as a whole.
# Our choice: L-BFGS capped at 20k evaluations (the baseline plateaued after
# ~16k of its 50k); the paper stops on its increment tolerance instead.
LBFGS_CAP = 20_000
LOG_DIR = "logs"

# Paper values (Tables 1 and 2, VP-NSFnet; each the BEST of ten runs), in %.
PAPER_ERRORS = {
    "4x50": (0.076, 0.412, 0.516), "7x50": (0.038, 0.255, 0.114),
    "7x100": (0.016, 0.183, 0.062), "10x100": (0.020, 0.115, 0.044),
    "10x200": (0.012, 0.103, 0.042), "10x250": (0.011, 0.101, 0.031),
    "10x300": (0.008, 0.072, 0.041),
    "alpha=1": (0.084, 0.425, 0.309), "alpha=100": (0.076, 0.412, 0.516),
}


def _exp(group, note, **changes):
    cfg = {"group": group, "note": note, "adam_schedule": PAPER_ADAM,
           "lbfgs_iters": LBFGS_CAP, "alpha": 100.0, "seed": 1234,
           "hidden_layers": 4, "width": 50, "n_interior": N_INTERIOR,
           "n_per_edge": N_PER_EDGE, "resume_from": None, "plateau": None}
    cfg.update(changes)
    return cfg


ADAM_20K = [(20_000, 1e-3)]     # our assumption for "20k Adam": constant lr 1e-3
# Listed roughly longest-first, so the parallel queue does not end on a long run.
EXPERIMENTS = {
    # A. architecture (paper Table 1 rows; 4x50 = ref)
    "A_10x300": _exp("A", "10x300", hidden_layers=10, width=300),
    "A_10x250": _exp("A", "10x250", hidden_layers=10, width=250),
    "A_10x200": _exp("A", "10x200", hidden_layers=10, width=200),
    "A_10x100": _exp("A", "10x100", hidden_layers=10, width=100),
    "A_7x100":  _exp("A", "7x100", hidden_layers=7, width=100),
    "A_7x50":   _exp("A", "7x50", hidden_layers=7, width=50),
    # C. number of training points
    "C_int10000": _exp("C", "10000 interior", n_interior=10_000),
    "C_int500":   _exp("C", "500 interior", n_interior=500),
    "C_bnd1000":  _exp("C", "1000 boundary", n_per_edge=251),
    "C_user_pts": _exp("C", "10000 interior + 1000 boundary", n_interior=10_000, n_per_edge=251),
    # B. boundary weight (alpha = 100 is ref; alpha = 1 is paper Table 2 and original code)
    "B_alpha1":    _exp("B", "alpha=1", alpha=1.0),
    "B_alpha10":   _exp("B", "alpha=10", alpha=10.0),
    "B_alpha1000": _exp("B", "alpha=1000", alpha=1000.0),
    # E. weight-initialisation seeds (training points stay fixed)
    **{f"E_seed{s}": _exp("E", f"seed {s}", seed=s) for s in (1, 2, 3, 4)},
    # D. optimiser budget
    "D_adam20k":       _exp("D", "20k Adam only", adam_schedule=ADAM_20K, lbfgs_iters=0),
    "D_adam20k_lbfgs": _exp("D", "20k Adam + L-BFGS", adam_schedule=ADAM_20K),
    "D_user_original": _exp("D", "user's original plan: 8x50, 10k/1k pts, 20k Adam, alpha=1",
                            hidden_layers=8, n_interior=10_000, n_per_edge=251,
                            adam_schedule=ADAM_20K, lbfgs_iters=0, alpha=1.0),
    # Reference: baseline's Adam stage (deterministic) + capped L-BFGS
    "ref": _exp("ref", "baseline, L-BFGS capped",
                resume_from=os.path.join(CKPT_DIR, "baseline_adam4.pt")),
}


# Converged architecture study (fixed L-BFGS, see train_lbfgs). Same Adam stage
# (resumed from each A_* run's Adam checkpoint; deterministic), then L-BFGS until
# the loss drops < 1% over 10k evaluations, cap 100k evaluations.
CONV_PLATEAU = (10_000, 0.01)
CONV_CAP = 100_000
for _name, (_h, _w) in {"ref": (4, 50), "A_7x50": (7, 50), "A_7x100": (7, 100),
                        "A_10x100": (10, 100), "A_10x200": (10, 200),
                        "A_10x250": (10, 250)}.items():
    # 10x300 dropped from the converged study (2026-11-06: fp64 L-BFGS ran out of memory on the
    # shared GPU). 10x200 ran later, alone (GPU memory). conv_10x250 is NOT rerun: its Stage 7 L-BFGS
    # (A_10x250) moved the weights and stopped on its own tolerance at 11.4k evals,
    # so A_10x250_final.pt is copied to conv_10x250_final.pt.
    _adam = "baseline_adam4.pt" if _name == "ref" else f"{_name}_adam4.pt"
    EXPERIMENTS["conv_" + ("4x50" if _name == "ref" else _name[2:])] = _exp(
        "conv", f"{_h}x{_w}", hidden_layers=_h, width=_w,
        lbfgs_iters=CONV_CAP, plateau=CONV_PLATEAU,
        resume_from=os.path.join(CKPT_DIR, _adam))

# Converged alpha study (added 2026-11-06 after review): B_alpha1 and B_alpha10 hit the
# same frozen-L-BFGS bug (their Stage 7 numbers are Adam-only), so all three alpha runs are
# redone from their Adam checkpoints with the fixed L-BFGS; compare with conv_4x50 (alpha=100).
for _a in (1.0, 10.0, 1000.0):
    EXPERIMENTS[f"convB_alpha{_a:g}"] = _exp(
        "convB", f"alpha={_a:g}", alpha=_a, lbfgs_iters=CONV_CAP, plateau=CONV_PLATEAU,
        resume_from=os.path.join(CKPT_DIR, f"B_alpha{_a:g}_adam4.pt"))


def run_named(name):
    cfg = {k: v for k, v in EXPERIMENTS[name].items() if k not in ("group", "note")}
    return run_experiment(name, **cfg)


def run_queue(names, workers):
    """Run experiments as separate processes, `workers` at a time.
    Skips those already finished (final checkpoint exists), so it can be restarted."""
    import subprocess
    os.makedirs(LOG_DIR, exist_ok=True)
    todo = [n for n in names if not os.path.exists(os.path.join(CKPT_DIR, f"{n}_final.pt"))]
    print(f"{len(names) - len(todo)} done already, {len(todo)} to run, {workers} at a time",
          flush=True)
    running, failed = {}, []
    while todo or running:
        while todo and len(running) < workers:
            name = todo.pop(0)
            log = open(os.path.join(LOG_DIR, f"{name}.log"), "w")
            running[name] = (subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "run", "--exp", name],
                stdout=log, stderr=subprocess.STDOUT), log, time.time())
            print(f"{time.strftime('%H:%M')} start {name}", flush=True)
        time.sleep(10)
        for name, (proc, log, t0) in list(running.items()):
            if proc.poll() is not None:
                log.close()
                del running[name]
                status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
                if proc.returncode:
                    failed.append(name)
                print(f"{time.strftime('%H:%M')} {status} {name} "
                      f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("all finished" + (f"; failed: {failed}" if failed else ""), flush=True)


def stage7_summary(path=os.path.join(FIG_DIR, "..", "stage7_summary.md")):
    """One markdown table per group from the final checkpoints (+ paper values)."""
    rows = {}
    for name in ["ref"] + [n for n in EXPERIMENTS if n != "ref"]:
        f = os.path.join(CKPT_DIR, f"{name}_final.pt")
        if os.path.exists(f):
            ck = torch.load(f, map_location="cpu", weights_only=False)
            rows[name] = (ck["info"], ck["history"][-1])
    ref_paper = {"A": "4x50", "B": "alpha=100"}
    lines = ["# Stage 7 results (relative L2 errors, %)\n"]
    for group, title in [("A", "Architecture"), ("B", "Boundary weight alpha"),
                         ("C", "Training points"), ("D", "Optimiser budget"),
                         ("E", "Initialisation seed"),
                         ("conv", "Architecture, converged L-BFGS (fixed; 'min' = L-BFGS only)"), ("convB", "Alpha, converged L-BFGS (vs conv_4x50)")]:
        lines += [f"\n## {group}. {title}\n",
                  "| run | eps_u | eps_v | eps_p | L_e | L_b | min | paper u / v / p |",
                  "|---|---|---|---|---|---|---|---|"]
        names = [n for n, c in EXPERIMENTS.items() if c["group"] == group]
        for name in (names if group.startswith("conv") else ["ref"] + names):
            if name not in rows:
                lines.append(f"| {name} | (not run yet) |||||||")
                continue
            info, last = rows[name]
            note = EXPERIMENTS[name]["note"] if name != "ref" else ref_paper.get(group, "")
            paper = PAPER_ERRORS.get(note if name != "ref" else ref_paper.get(group, ""))
            paper = " / ".join(f"{e:.3f}" for e in paper) if paper else ""
            lines.append(
                f"| {name} | {100 * info['eps_u']:.3f} | {100 * info['eps_v']:.3f} | "
                f"{100 * info['eps_p']:.3f} | {last['L_e']:.2e} | {last['L_b']:.2e} | "
                f"{info['train_seconds'] / 60:.0f} | {paper} |")
    seeds = [rows[n][0] for n in ["ref", "E_seed1", "E_seed2", "E_seed3", "E_seed4"] if n in rows]
    if len(seeds) > 1:
        lines.append(f"\nSeed spread over {len(seeds)} runs (mean ± std, %):")
        for k in ("eps_u", "eps_v", "eps_p"):
            t = torch.tensor([100 * s[k] for s in seeds])
            lines.append(f"- {k}: {t.mean():.3f} ± {t.std():.3f} (min {t.min():.3f}, max {t.max():.3f})")
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    print(text)


# %% Stage 7 figures (fig5, 5b, 5c, 6-8), from the final checkpoints of the Stage 7 runs
MARKERS = {"u": "o", "v": "s", "p": "^"}      # second cue beside colour (v/p are close for deutans)


def _n_params(layers, width, n_in=2, n_out=3):
    return (n_in + 1) * width + (layers - 1) * (width + 1) * width + (width + 1) * n_out


def _stage7_rows():
    rows = {}
    for name in EXPERIMENTS:
        f = os.path.join(CKPT_DIR, f"{name}_final.pt")
        if os.path.exists(f):
            ck = torch.load(f, map_location="cpu", weights_only=False)
            rows[name] = {"info": ck["info"], "history": ck["history"]}
    missing = [n for n in EXPERIMENTS if n not in rows]
    if missing:
        raise SystemExit(f"Stage 7 runs missing final checkpoints: {missing}")
    return rows


def _lbfgs_evals(history):
    lb = [r["step"] for r in history if r["phase"] == "lbfgs"]
    return (lb[-1] - lb[0] + 100) if lb else 0      # rows every 100 evaluations


def _err_legend(ax, ours="this work", paper="paper (best of 10)", **kw):
    """Legend: one entry per variable (colour + marker), then the ours/paper styles in grey."""
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=SERIES[n], marker=MARKERS[n], ms=5, lw=1.2, label=f"$\\varepsilon_{n}$")
         for n in "uvp"]
    h += [Line2D([], [], color="0.35", marker="o", ms=5, lw=1.2, label=ours),
          Line2D([], [], color="0.35", marker="o", ms=5, mfc="white", lw=0.9, ls="--", label=paper)]
    ax.legend(handles=h, **kw)


def fig_architecture(rows):
    """Final errors vs. trainable parameters, ours (filled) and paper Table 1 (open)."""
    names = ["ref", "A_7x50", "A_7x100", "A_10x100", "A_10x200", "A_10x250", "A_10x300"]
    arch = {n: (rows[n]["info"]["hidden_layers"], rows[n]["info"]["width"]) for n in names}
    order = sorted(names, key=lambda n: _n_params(*arch[n]))
    x = list(range(len(order)))                  # evenly spaced, ordered by parameter count
    labels = [f"{l}$\\times${w} ({_n_params(l, w) / 1e3:.0f}k)" for l, w in (arch[n] for n in order)]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.7), constrained_layout=True, sharey=True)
    for k, (ax, var) in enumerate(zip(axes, "uvp")):
        key = f"eps_{var}"
        ours = [100 * rows[n]["info"][key] for n in order]
        paper = [PAPER_ERRORS[f"{l}x{w}"]["uvp".index(var)] for l, w in (arch[n] for n in order)]
        c, m = SERIES[var], MARKERS[var]
        ax.plot(x, paper, color=c, ls="--", lw=0.9, marker=m, ms=5, mfc="white", zorder=2)
        ax.plot(x, ours, color=c, lw=1.2, marker=m, ms=5, mec="white", mew=0.6, zorder=3)
        # the one run whose L-BFGS stopped on its own tolerance (all others hit the 20k cap)
        for xi, yi, n in zip(x, ours, order):
            if _lbfgs_evals(rows[n]["history"]) < LBFGS_CAP - 100:
                ax.plot(xi, yi, ls="none", marker="o", ms=11, mfc="none", mec="0.2", mew=0.8,
                        zorder=4)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_title(f"$\\varepsilon_{var}$")
        ax.grid(alpha=0.25, lw=0.5)
        _panel(ax, "abc"[k])
    axes[0].set_ylabel(r"relative $L_2$ error (%)")
    axes[1].set_xlabel("network: hidden layers $\\times$ width (trainable parameters)")
    from matplotlib.lines import Line2D
    _err_legend(fig, loc="outside lower center", ncol=6)
    axes[0].legend(handles=[Line2D([], [], ls="none", marker="o", ms=11, mfc="none", mec="0.2",
                                   mew=0.8, label="L-BFGS converged\n(others hit 20k cap)")],
                   loc="upper right", fontsize=7, handletextpad=0.3)
    _save(fig, "fig5_architecture")


def fig_architecture_converged():
    """fig5b: errors vs. network size after the L-BFGS fix (conv_* runs, trained to the plateau stop),
    next to paper Table 1 (open). Runs whose conv_*
    checkpoint is not there yet (still training) are simply left out of the converged line."""
    _paper_style()
    sizes = [(4, 50), (7, 50), (7, 100), (10, 100), (10, 200), (10, 250)]    # 10x300 dropped (see conv)
    order = sorted(sizes, key=lambda lw: _n_params(*lw))
    x = list(range(len(order)))
    labels = [f"{l}$\\times${w} ({_n_params(l, w) / 1e3:.0f}k)" for l, w in order]

    def info(name):
        f = os.path.join(CKPT_DIR, f"{name}_final.pt")
        return torch.load(f, map_location="cpu", weights_only=False)["info"] if os.path.exists(f) else None
    conv = [info(f"conv_{l}x{w}") for l, w in order]
    pending = [f"{l}$\\times${w}" for (l, w), c in zip(order, conv) if c is None]

    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.7), constrained_layout=True, sharey=True)
    for k, (ax, var) in enumerate(zip(axes, "uvp")):
        key, c, m = f"eps_{var}", SERIES[var], MARKERS[var]
        paper = [PAPER_ERRORS[f"{l}x{w}"]["uvp".index(var)] for l, w in order]
        ax.plot(x, paper, color=c, ls="--", lw=0.9, marker=m, ms=5, mfc="white", zorder=2)
        yv = [100 * i[key] if i else float("nan") for i in conv]          # NaN breaks the line at pending runs
        ax.plot(x, yv, color=c, lw=1.6, marker=m, ms=6, mec="white", mew=0.6, zorder=3)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=40, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_title(f"$\\varepsilon_{var}$  ({'circles' if var == 'u' else 'squares' if var == 'v' else 'triangles'})")
        ax.grid(alpha=0.25, lw=0.5)
        _panel(ax, "abc"[k])
    axes[0].set_ylabel(r"relative $L_2$ error (%)")
    axes[1].set_xlabel("network: hidden layers $\\times$ width (trainable parameters)")
    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple
    def trio(**kw):            # one legend entry showing the u, v, p markers in their colours
        return tuple(Line2D([], [], color=SERIES[n], marker=MARKERS[n], **kw) for n in "uvp")
    h = [trio(ms=6, lw=1.6, mec="white", mew=0.6),
         trio(ms=5, lw=0.9, ls="--", mfc="white")]
    lab = ["ours, converged L-BFGS (filled)", "paper Table 1, best of 10 (hollow)"]
    leg = fig.legend(h, lab, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)}, handlelength=4.5,
                     loc="outside lower center", ncol=2)
    if pending:
        print("  converged run not finished yet (left out):", ", ".join(p.replace("$\\times$", "x") for p in pending))
    _save(fig, "fig5b_architecture_converged")


def fig_ablation_overview():
    """fig5c: one-slide summary of Stage 7 groups B-E. Each row = one run; x = its error divided
    by the baseline (ref) error, log scale; grey band = seed spread (min-max over 5 seeds)."""
    _paper_style()
    def info(name):
        return torch.load(os.path.join(CKPT_DIR, f"{name}_final.pt"), map_location="cpu",
                          weights_only=False)["info"]
    ref = info("ref")
    runs = [("Boundary weight", [("convB_alpha1", r"$\alpha = 1$"), ("convB_alpha10", r"$\alpha = 10$"),
                                 ("convB_alpha1000", r"$\alpha = 1000$")]),   # vs conv_4x50
            ("Optimiser", [("D_adam20k", "20k Adam only"), ("D_adam20k_lbfgs", "20k Adam + L-BFGS"),
                           ("D_user_original", "my first plan (Adam only)")]),
            ("Training points", [("C_int500", "500 interior"), ("C_int10000", "10 000 interior"),
                                 ("C_bnd1000", "1000 boundary")])]
    seeds = [ref] + [info(f"E_seed{k}") for k in range(1, 5)]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0), constrained_layout=True, sharey=True,
                             sharex=True)
    labels, ypos, y = [], [], 0
    group_y = []
    for g, items in runs:
        group_y.append((g, y))
        for name, lab in items:
            labels.append(lab); ypos.append(y); y += 1
        y += 0.6
    for k, (ax, var) in enumerate(zip(axes, "uvp")):
        key = f"eps_{var}"
        s = [i[key] / ref[key] for i in seeds]
        ax.axvspan(min(s), max(s), color="0.85", lw=0, zorder=0)
        ax.axvline(1, color="0.4", lw=0.8, zorder=1)
        xs = [info(n)[key] / (info("conv_4x50") if n.startswith("conv") else ref)[key] for _, it in runs for n, _ in it]
        ax.plot(xs, ypos, ls="none", marker=MARKERS[var], ms=6, color=SERIES[var], mec="white",
                mew=0.5, zorder=3)
        ax.set_xscale("log")
        ax.set_title(f"$\\varepsilon_{var}$ / baseline")
        ax.grid(alpha=0.25, lw=0.5, axis="x", which="both")
    axes[0].set_xlim(0.4, 150)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].invert_yaxis()
    for g, gy in group_y:
        axes[0].annotate(g, (0, gy - 0.55), xycoords=("axes fraction", "data"), fontsize=8,
                         fontweight="bold", ha="right", va="bottom", xytext=(-4, 0),
                         textcoords="offset points")
    axes[1].set_xlabel("error relative to baseline (1 = same; grey = spread over 5 seeds)")
    _save(fig, "fig5c_ablation_overview")


def fig_alpha(rows):
    """(a) final errors vs. boundary weight alpha; (b) final unweighted loss terms vs. alpha."""
    names = ["convB_alpha1", "convB_alpha10", "conv_4x50", "convB_alpha1000"]   # converged L-BFGS
    alphas = [rows[n]["info"]["alpha"] for n in names]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 2.8), constrained_layout=True)
    for var in "uvp":
        key = f"eps_{var}"
        c, m = SERIES[var], MARKERS[var]
        a.plot(alphas, [100 * rows[n]["info"][key] for n in names], color=c, marker=m, ms=5,
               mec="white", mew=0.6, lw=1.2, zorder=3)
        pa = [1.0, 100.0]
        a.plot(pa, [PAPER_ERRORS[f"alpha={int(p)}"]["uvp".index(var)] for p in pa], color=c,
               marker=m, ms=5, mfc="white", ls="none", zorder=2)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xticks(alphas)
    a.set_xlabel(r"boundary weight $\alpha$")
    a.set_ylabel(r"relative $L_2$ error (%)")
    a.grid(alpha=0.25, lw=0.5)
    _err_legend(a, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=5, fontsize=7,
                paper="paper (best of 10)", columnspacing=1.0, handlelength=1.6)
    _panel(a, "a")

    for key, label, col, m in (("L_e", r"$\mathcal{L}_e$ (PDE residual)", "#e07a1f", "D"),
                               ("L_b", r"$\mathcal{L}_b$ (boundary, unweighted)", "#2a9d8f", "v")):
        b.plot(alphas, [rows[n]["history"][-1][key] for n in names], color=col, marker=m, ms=5,
               mec="white", mew=0.6, lw=1.2, label=label)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xticks(alphas)
    b.set_xlabel(r"boundary weight $\alpha$")
    b.set_ylabel("final loss term")
    b.grid(alpha=0.25, lw=0.5)
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=7)
    _panel(b, "b")
    _save(fig, "fig6_alpha")


def fig_optimiser(rows):
    """Total loss and eps_v vs. iteration for the optimiser-budget runs."""
    runs = [("ref", "paper schedule: 110k Adam + L-BFGS (4$\\times$50, $\\alpha$=100)", "#222222", "-"),
            ("D_adam20k_lbfgs", "20k Adam + L-BFGS (4$\\times$50, $\\alpha$=100)", "#7b4fa0", "-"),
            ("D_user_original", "20k Adam only (8$\\times$50, $\\alpha$=1, 10k/1k points)", "#c47a12", "-")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(FULL_W, 3.0), constrained_layout=True)
    for name, label, col, ls in runs:
        h = rows[name]["history"]
        s = [r["step"] / 1e3 for r in h]
        a.semilogy(s, [r["total"] for r in h], color=col, ls=ls, lw=0.9, label=label)
        ev = [r for r in h if "eps_v" in r]
        b.semilogy([r["step"] / 1e3 for r in ev], [100 * r["eps_v"] for r in ev], color=col,
                   ls=ls, lw=1.1)
        # mark where L-BFGS takes over
        lb = [r for r in h if r["phase"] == "lbfgs"]
        if lb:
            for ax in (a, b):
                ax.axvline(lb[0]["step"] / 1e3, color=col, lw=0.6, ls=":", alpha=0.8)
    for ax in (a, b):
        ax.set_xlabel(r"iteration ($\times 10^3$)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(0, None)
    a.set_ylabel(r"total loss $\mathcal{L}$")
    b.set_ylabel(r"$\varepsilon_v$ (%)")
    b.text(0.98, 0.97, "dotted: start of L-BFGS\n(L-BFGS counted in loss evaluations)", transform=b.transAxes, ha="right", va="top",
           fontsize=7, color="0.35")
    fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=1)
    _panel(a, "a")
    _panel(b, "b")
    _save(fig, "fig7_optimiser")


def fig_points_seeds(rows):
    """Training-point runs against the seed-to-seed spread (5 seeds incl. ref)."""
    import numpy as np
    seeds = ["ref", "E_seed1", "E_seed2", "E_seed3", "E_seed4"]
    cases = [("C_int500", "500 / 400"), ("ref", "2601 / 400 (ref)"), ("C_bnd1000", "2601 / 1000"),
             ("C_int10000", "10000 / 400"), ("C_user_pts", "10000 / 1000")]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.4), constrained_layout=True, sharey=True)
    ypos = np.arange(len(cases))[::-1]
    for k, (ax, var) in enumerate(zip(axes, "uvp")):
        key = f"eps_{var}"
        sv = np.array([100 * rows[n]["info"][key] for n in seeds])
        mu, sd = sv.mean(), sv.std(ddof=1)
        ax.axvspan(mu - 2 * sd, mu + 2 * sd, color="0.85", lw=0, zorder=0)
        ax.axvline(mu, color="0.55", lw=0.7, zorder=1)
        ax.plot(sv, np.full_like(sv, ypos.max() + 1), ls="none", marker=MARKERS[var], ms=4,
                color="0.5", zorder=2)
        ax.plot([100 * rows[n]["info"][key] for n, _ in cases], ypos, ls="none",
                marker=MARKERS[var], ms=6, color=SERIES[var], mec="white", mew=0.6, zorder=3)
        ax.set_title(f"$\\varepsilon_{var}$ (%)")
        ax.grid(axis="x", alpha=0.25, lw=0.5)
        ax.set_xlim(0, None)
        _panel(ax, "abc"[k])
    axes[0].set_yticks(list(ypos) + [ypos.max() + 1])
    axes[0].set_yticklabels([c[1] for c in cases] + ["5 init. seeds"])
    axes[0].set_ylabel("interior / boundary points")
    axes[1].text(0.5, -0.22, r"grey band: seed mean $\pm$ 2 s.d.", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=7, color="0.35")
    _save(fig, "fig8_points_seeds")


def make_stage7_figures():
    _paper_style()
    rows = _stage7_rows()
    print(f"Stage 7 figures -> {PAPER_FIG_DIR}/")
    fig_architecture(rows)
    fig_alpha(rows)
    fig_optimiser(rows)
    fig_points_seeds(rows)
    fig_architecture_converged()    # fig5b: converged-L-BFGS architecture study
    fig_ablation_overview()         # fig5c: one-panel overview of groups B-D


# %% Heat-transfer extension H1: formulation (our own extension, not in the paper)
# Steady forced convection of a passive temperature field T(x, y) carried by the
# Kovasznay flow (one-way coupling: T does not affect u, v, p, so the exact flow
# stays valid). Nondimensional energy equation, no source, no viscous heating:
#     u T_x + v T_y = (1/Pe) (T_xx + T_yy),      Pe = Re * Pr
# Boundary conditions on [-0.5, 1] x [-0.5, 1.5]:
#   inlet  x = -0.5 : T = T_in(y) = (1 - cos(pi (y + 0.5))) / 2   (hot core, 0 at y edges)
#   y = -0.5, 1.5   : T = 0   (cold isothermal walls; v = 0 there, so they are
#                              streamlines: no flow crosses them, like channel walls)
#   outlet x = 1    : T_x = 0 (standard outflow condition)
# T_in is smooth and matches the wall value at the corners, so T is continuous.
# Physics: the hot core is advected downstream and cooled by the walls; thermal
# boundary layers at the walls thin as Pe grows (~ Pe^-1/2) -> harder for the PINN.
# Pe values: Re = 40 with Pr = 0.7 (air) -> 28, Pr = 7 (water) -> 280; plus 1, 10, 100.
# No exact solution exists -> reference from a fine finite-difference solve (H2).
PE_LIST = (1.0, 10.0, 28.0, 100.0, 280.0)
HEAT_DIR = "heat"


def inlet_temperature(y):
    """T_in(y) on x = X_MIN; torch tensor in, same shape out."""
    return 0.5 * (1.0 - torch.cos(math.pi * (y - Y_MIN)))


# %% Heat-transfer extension H2: finite-difference reference solution
def solve_heat_fd(pe, n):
    """Second-order central differences on an n x n node grid (exact u, v), direct
    sparse solve. Returns T as an (n, n) float64 numpy array, rows = y, cols = x
    (same layout as validation_grid). Outlet T_x = 0 via a mirrored ghost node."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    xs = np.linspace(X_MIN, X_MAX, n)
    ys = np.linspace(Y_MIN, Y_MAX, n)
    hx, hy = xs[1] - xs[0], ys[1] - ys[0]
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    u, v, _ = exact_solution(torch.from_numpy(X), torch.from_numpy(Y))
    u, v = u.numpy(), v.numpy()

    idx = np.arange(n * n).reshape(n, n)          # node (j, i) -> row j*n + i
    J, I = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    wall = (J == 0) | (J == n - 1)
    inlet = (I == 0) & ~wall
    inner = ~wall & ~inlet                        # interior + outlet column

    rows, cols, vals = [], [], []
    rhs = np.zeros(n * n)
    # Dirichlet rows
    for mask in (wall, inlet):
        k = idx[mask]
        rows.append(k); cols.append(k); vals.append(np.ones(k.size))
    rhs[idx[inlet]] = inlet_temperature(torch.from_numpy(Y[inlet])).numpy()
    # u T_x + v T_y - (T_xx + T_yy)/Pe = 0
    j, i = J[inner], I[inner]
    uu, vv = u[inner], v[inner]
    dx2, dy2 = 1.0 / (pe * hx**2), 1.0 / (pe * hy**2)
    a_e = uu / (2 * hx) - dx2
    a_w = -uu / (2 * hx) - dx2
    a_n = vv / (2 * hy) - dy2
    a_s = -vv / (2 * hy) - dy2
    a_p = np.full(j.size, 2 * dx2 + 2 * dy2)
    out = i == n - 1                              # ghost T_{n} = T_{n-2}: east -> west
    a_w = np.where(out, a_w + a_e, a_w)
    e_col = np.where(out, i, i + 1)               # placeholder for outlet (coef 0 below)
    a_e = np.where(out, 0.0, a_e)
    k = idx[j, i]
    for col, a in ((idx[j, i], a_p), (idx[j, e_col], a_e), (idx[j, i - 1], a_w),
                   (idx[j + 1, i], a_n), (idx[j - 1, i], a_s)):
        rows.append(k); cols.append(col); vals.append(a)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n * n, n * n))       # duplicates (outlet) are summed
    return spla.spsolve(A.tocsc(), rhs).reshape(n, n)


def heat_reference(pe_list=PE_LIST, levels=(401, 801, 1601), n_out=N_VAL):
    """FD solves on 3 grids per Pe; reference = finest, sampled onto the 201 x 201
    validation grid. Grid-convergence check (Richardson): observed order
    q = log2(|T_coarse - T_mid| / |T_mid - T_fine|) on the common 201 x 201 nodes,
    estimated relative error of the reference = |T_mid - T_fine| / (2^q - 1) / |T_fine|."""
    import numpy as np
    os.makedirs(HEAT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    report = []
    for pe in pe_list:
        sols, t0 = [], time.time()
        for n in levels:
            step = (n - 1) // (n_out - 1)
            sols.append(solve_heat_fd(pe, n)[::step, ::step])
        c, m, f = sols
        d1, d2 = np.linalg.norm(c - m), np.linalg.norm(m - f)
        q = math.log2(d1 / d2)
        err = d2 / (2**q - 1) / np.linalg.norm(f)
        info = {"pe": pe, "n_fd": levels[-1], "order": q, "rel_err_est": err,
                "seconds": time.time() - t0}
        torch.save({"T": torch.from_numpy(f), **info},
                   os.path.join(HEAT_DIR, f"ref_Pe{pe:g}.pt"))
        line = (f"Pe {pe:>5g}: observed order {q:.2f}, est. rel. error of reference "
                f"{err:.1e}, T range [{f.min():.3f}, {f.max():.3f}]  ({info['seconds']:.0f} s)")
        print(line, flush=True)
        report.append(line)
    plot_heat_reference(pe_list)
    return report


def plot_heat_reference(pe_list=PE_LIST):
    val = validation_grid()
    X, Y = val["X"].numpy(), val["Y"].numpy()
    fig, axes = plt.subplots(1, len(pe_list), figsize=(3.2 * len(pe_list), 4.2),
                             sharey=True, constrained_layout=True)
    for ax, pe in zip(axes, pe_list):
        T = torch.load(os.path.join(HEAT_DIR, f"ref_Pe{pe:g}.pt"))["T"].numpy()
        cf = ax.contourf(X, Y, T, levels=torch.linspace(0, 1, 21).numpy(), cmap="inferno")
        ax.streamplot(X, Y, val["u"].reshape(X.shape).numpy(),
                      val["v"].reshape(X.shape).numpy(), color="w", linewidth=0.4,
                      density=0.6, arrowsize=0.5)
        ax.set_title(f"Pe = {pe:g}")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(cf, ax=axes, shrink=0.8, label="T")
    fig.savefig(os.path.join(FIG_DIR, "heat_reference.png"), dpi=150)
    plt.close(fig)


# %% Heat-transfer extension H3: one network (x, y) -> (u, v, p, T) and its loss
# Our choices: same point sets as the flow problem (2601 interior, the 400
# boundary points split by edge), same alpha = 100 for every boundary term, and
# the flow is learnt jointly (not taken from the exact solution).
#   L = L_e + L_T + alpha * (L_b + L_bT)
#   L_T  = mean r_T^2,  r_T = u T_x + v T_y - (T_xx + T_yy) / Pe
#   L_bT = mean (T - T_bc)^2 on inlet + walls  +  mean T_x^2 on the outlet
def heat_boundary_points(n_per_edge=N_PER_EDGE):
    """Same 400 points as boundary_points (order: left, right, bottom, top).
    Returns flow BC tuple (x, y, u, v) and temperature BC dict."""
    x, y, u, v = boundary_points(n_per_edge)
    m = n_per_edge - 1
    right = torch.zeros(4 * m, dtype=torch.bool)
    right[m:2 * m] = True
    left = torch.zeros_like(right)
    left[:m] = True
    T_bc = torch.where(left.reshape(-1, 1), inlet_temperature(y), torch.zeros_like(y))
    heat = {"xd": x[~right], "yd": y[~right], "Td": T_bc[~right],     # Dirichlet: 300 pts
            "xn": x[right], "yn": y[right]}                              # outlet T_x = 0: 100 pts
    return (x, y, u, v), heat


def heat_residuals(model, x, y, pe):
    """r_c, r_u, r_v (flow) and r_T (energy) from one forward pass of a 4-output net."""
    x = x.detach().requires_grad_(True)
    y = y.detach().requires_grad_(True)
    u, v, p, T = model(x, y)
    u_x, u_y = grad(u, x), grad(u, y)
    v_x, v_y = grad(v, x), grad(v, y)
    p_x, p_y = grad(p, x), grad(p, y)
    T_x, T_y = grad(T, x), grad(T, y)
    u_xx, u_yy = grad(u_x, x), grad(u_y, y)
    v_xx, v_yy = grad(v_x, x), grad(v_y, y)
    T_xx, T_yy = grad(T_x, x), grad(T_y, y)
    r_c = u_x + v_y
    r_u = u * u_x + v * u_y + p_x - (u_xx + u_yy) / RE
    r_v = u * v_x + v * v_y + p_y - (v_xx + v_yy) / RE
    r_T = u * T_x + v * T_y - (T_xx + T_yy) / pe
    return r_c, r_u, r_v, r_T


def heat_loss(model, interior, boundary, heat_bc, pe, alpha=100.0):
    r_c, r_u, r_v, r_T = heat_residuals(model, *interior, pe)
    L_e = r_c.pow(2).mean() + r_u.pow(2).mean() + r_v.pow(2).mean()
    L_T = r_T.pow(2).mean()
    xb, yb, ub, vb = boundary
    u_b, v_b, _, _ = model(xb, yb)
    L_b = (u_b - ub).pow(2).mean() + (v_b - vb).pow(2).mean()
    T_d = model(heat_bc["xd"], heat_bc["yd"])[3]
    xn = heat_bc["xn"].detach().requires_grad_(True)
    T_n = model(xn, heat_bc["yn"])[3]
    L_bT = (T_d - heat_bc["Td"]).pow(2).mean() + grad(T_n, xn).pow(2).mean()
    loss = L_e + L_T + alpha * (L_b + L_bT)
    parts = {"total": loss, "L_e": L_e, "L_T": L_T, "L_b": L_b, "L_bT": L_bT}
    return loss, {k: t.detach() for k, t in parts.items()}


def heat_errors(model, val):
    """Flow errors as before + eps_T against the FD reference (val["T"])."""
    with torch.no_grad():
        u, v, p, T = model(val["x"], val["y"])
    p = p - p.mean() + val["p"].mean()
    rel = lambda a, b: (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()
    return {"eps_u": rel(u, val["u"]), "eps_v": rel(v, val["v"]),
            "eps_p": rel(p, val["p"]), "eps_T": rel(T, val["T"])}


# %% Heat-transfer extension H4: training (same optimiser recipe as the flow baseline)
def run_heat(tag, pe, adam_schedule=PAPER_ADAM, lbfgs_iters=CONV_CAP, plateau=CONV_PLATEAU,
             alpha=100.0, seed=1234, hidden_layers=4, width=50):
    os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    ref = torch.load(os.path.join(HEAT_DIR, f"ref_Pe{pe:g}.pt"))
    interior = tuple(t.to(DEVICE) for t in interior_points(N_INTERIOR))
    boundary, heat_bc = heat_boundary_points(N_PER_EDGE)
    boundary = tuple(t.to(DEVICE) for t in boundary)
    heat_bc = {k: t.to(DEVICE) for k, t in heat_bc.items()}
    val = validation_grid()
    val["T"] = ref["T"].float().reshape(-1, 1)           # same (rows y, cols x) layout
    val = {k: t.to(DEVICE) for k, t in val.items()}

    torch.manual_seed(seed)
    model = VPNet(hidden_layers, width, n_out=4).to(DEVICE)
    info = {"tag": tag, "pe": pe, "adam_schedule": adam_schedule, "lbfgs_iters": lbfgs_iters,
            "plateau": plateau, "alpha": alpha, "seed": seed,
            "hidden_layers": hidden_layers, "width": width, "n_out": 4}
    print(f"=== {tag}: {info} on {DEVICE} ===", flush=True)
    data = {"interior": interior, "boundary": boundary, "heat_bc": heat_bc, "val": val}
    loss_fn = lambda: heat_loss(model, data["interior"], data["boundary"], data["heat_bc"],
                                pe, alpha)
    error_fn = lambda: heat_errors(model, data["val"])

    t0, history = time.time(), []
    step = train_adam(model, None, None, None, adam_schedule, alpha, history, tag=tag,
                      loss_fn=loss_fn, error_fn=error_fn)
    if lbfgs_iters > 0:                                  # float64 L-BFGS, as for the flow
        model.double()
        data["interior"] = tuple(t.double() for t in interior)
        data["boundary"] = tuple(t.double() for t in boundary)
        data["heat_bc"] = {k: t.double() for k, t in heat_bc.items()}
        data["val"] = {k: t.double() for k, t in val.items()}
        step = train_lbfgs(model, None, None, None, alpha, history, step,
                           max_iter=lbfgs_iters, plateau=plateau,
                           loss_fn=loss_fn, error_fn=error_fn)
    info["train_seconds"] = time.time() - t0
    _, parts = loss_fn()
    errors = error_fn()
    _record(history, step, history[-1]["phase"], parts, errors)
    info.update(errors)
    save_checkpoint(os.path.join(CKPT_DIR, f"{tag}_final.pt"), model, history, info)
    plot_history(history, os.path.join(FIG_DIR, f"{tag}_history.png"), title=tag)
    print(f"Final: eps_u {100 * errors['eps_u']:.3f}%  eps_v {100 * errors['eps_v']:.3f}%  "
          f"eps_p {100 * errors['eps_p']:.3f}%  eps_T {100 * errors['eps_T']:.3f}%  "
          f"({info['train_seconds'] / 60:.1f} min)", flush=True)
    return model, history, info


# %% Heat-transfer extension H5: analysis (eps_T vs Pe, wall heat flux / Nusselt, energy balance)
# Wall heat flux from the fluid into the cold walls (T_w = 0), per unit (1/Pe):
#   q_bot(x) = +T_y(x, -0.5),   q_top(x) = -T_y(x, 1.5)
# Local Nusselt number (our definition): Nu(x) = H q(x) / dT_ref with H = 2 (wall
# spacing) and dT_ref = 1 (peak inlet T minus wall T). No bulk temperature: the
# Kovasznay flow reverses (u < 0) near the inlet, so a u-weighted mean T is ill-posed.
# Global energy balance (div(u T - grad T / Pe) = 0 because div u = 0):
#   Q_in   = int_inlet  (u T - T_x / Pe) dy
#   Q_out  = int_outlet  u T dy                      (T_x = 0 there)
#   Q_wall = (1/Pe) int (q_bot + q_top) dx,          Q_in = Q_out + Q_wall
# The PINN never sees this integral law, so its imbalance is an independent check.
HEAT_FIG_N = 1601                       # points per line for the PINN wall/edge integrals


def _trapz(f, h):
    return h * (f.sum() - 0.5 * (f[0] + f[-1]))


def _fd_wall_data(T, pe):
    """Wall flux profiles and energy-balance terms from one FD solution (n x n, rows y).
    One-sided 3-point derivatives (2nd order, as the scheme). Profiles on all n x-nodes."""
    import numpy as np
    n = T.shape[0]
    xs, ys = np.linspace(X_MIN, X_MAX, n), np.linspace(Y_MIN, Y_MAX, n)
    hx, hy = xs[1] - xs[0], ys[1] - ys[0]
    q_bot = (-3 * T[0] + 4 * T[1] - T[2]) / (2 * hy)
    q_top = (3 * T[-1] - 4 * T[-2] + T[-3]) / (-2 * hy)
    T_x_in = (-3 * T[:, 0] + 4 * T[:, 1] - T[:, 2]) / (2 * hx)
    u_in, _, _ = exact_solution(torch.full((n,), X_MIN, dtype=torch.float64), torch.from_numpy(ys))
    u_out, _, _ = exact_solution(torch.full((n,), X_MAX, dtype=torch.float64), torch.from_numpy(ys))
    Q_in = _trapz(u_in.numpy() * T[:, 0] - T_x_in / pe, hy)
    Q_out = _trapz(u_out.numpy() * T[:, -1], hy)
    Q_wall = _trapz(q_bot + q_top, hx) / pe
    return {"q_bot": q_bot, "q_top": q_top, "Q_in": Q_in, "Q_out": Q_out, "Q_wall": Q_wall}


def heat_wall_reference(pe, levels=(801, 1601), n_out=N_VAL):
    """Reference wall flux / energy terms: FD on two grids + Richardson (f = (4 f_fine -
    f_coarse) / 3), profiles sampled on the n_out validation x-nodes. Cached in HEAT_DIR."""
    import numpy as np
    path = os.path.join(HEAT_DIR, f"ref_wall_Pe{pe:g}.pt")
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    t0 = time.time()
    c, f = (_fd_wall_data(solve_heat_fd(pe, n), pe) for n in levels)
    sc, sf = ((n - 1) // (n_out - 1) for n in levels)
    out, err = {}, {}
    for k in c:
        a, b = (c[k][::sc], f[k][::sf]) if k.startswith("q_") else (c[k], f[k])
        out[k] = (4 * b - a) / 3
        err[k] = np.max(np.abs(b - a)) / 3 / np.max(np.abs(out[k]))
    out.update({"x": np.linspace(X_MIN, X_MAX, n_out), "rel_err_est": err,
                "seconds": time.time() - t0})
    torch.save(out, path)
    print(f"  wall reference Pe {pe:g}: est. rel. errors "
          + ", ".join(f"{k} {v:.1e}" for k, v in err.items()) + f" ({out['seconds']:.0f} s)",
          flush=True)
    return out


def _heat_ckpt(pe):
    """Final checkpoint if training finished, else the latest Adam phase checkpoint."""
    final = os.path.join(CKPT_DIR, f"heat_Pe{pe:g}_final.pt")
    if os.path.exists(final):
        return final, "final"
    for k in (4, 3, 2, 1):
        p = os.path.join(CKPT_DIR, f"heat_Pe{pe:g}_adam{k}.pt")
        if os.path.exists(p):
            return p, f"adam{k} (NOT final)"
    return None, None


def load_heat_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    info = ckpt["info"]
    model = VPNet(info.get("hidden_layers", 4), info.get("width", 50),
                  n_out=info.get("n_out", 4)).to(DEVICE).double()
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["history"], info


def _line(a, b, fixed, n, along_x):
    s = torch.linspace(a, b, n, dtype=torch.float64, device=DEVICE).reshape(-1, 1)
    c = torch.full_like(s, fixed)
    return (s, c) if along_x else (c, s)


def heat_pinn_wall_data(model, pe, n=HEAT_FIG_N):
    """Same quantities as _fd_wall_data, from the PINN by autograd (float64)."""
    h_x, h_y = (X_MAX - X_MIN) / (n - 1), (Y_MAX - Y_MIN) / (n - 1)
    out = {}
    for name, fixed, sign in (("q_bot", Y_MIN, 1.0), ("q_top", Y_MAX, -1.0)):
        x, y = _line(X_MIN, X_MAX, fixed, n, along_x=True)
        y.requires_grad_(True)
        T = model(x, y)[3]
        out[name] = (sign * grad(T, y)).detach().cpu().numpy().ravel()
    x, y = _line(Y_MIN, Y_MAX, X_MIN, n, along_x=False)
    x.requires_grad_(True)
    u, _, _, T = model(x, y)
    flux_in = (u * T - grad(T, x) / pe).detach().cpu().numpy().ravel()
    x, y = _line(Y_MIN, Y_MAX, X_MAX, n, along_x=False)
    with torch.no_grad():
        u, _, _, T = model(x, y)
    out["Q_in"] = _trapz(flux_in, h_y)
    out["Q_out"] = _trapz((u * T).cpu().numpy().ravel(), h_y)
    out["Q_wall"] = _trapz(out["q_bot"] + out["q_top"], h_x) / pe
    out["x"] = torch.linspace(X_MIN, X_MAX, n, dtype=torch.float64).numpy()
    return out


def heat_analysis_one(pe):
    """All H5 numbers for one Pe. Returns (row dict, fields for plotting)."""
    import numpy as np
    path, label = _heat_ckpt(pe)
    if path is None:
        return None, None
    model, history, info = load_heat_model(path)
    ref = torch.load(os.path.join(HEAT_DIR, f"ref_Pe{pe:g}.pt"))
    val = validation_grid()
    val["T"] = ref["T"].reshape(-1, 1)
    val = {k: t.to(DEVICE).double() for k, t in val.items()}
    errs = heat_errors(model, val)
    n = val["X"].shape[0]
    with torch.no_grad():
        T = model(val["x"], val["y"])[3].reshape(n, n).cpu().numpy()
    T_ref = val["T"].reshape(n, n).cpu().numpy()
    Y = val["Y"].cpu().numpy()
    j, i = np.unravel_index(np.abs(T - T_ref).argmax(), T.shape)
    r_T = heat_residuals(model, val["x"], val["y"], pe)[3].detach()

    wall_ref, wall = heat_wall_reference(pe), heat_pinn_wall_data(model, pe)
    step = (HEAT_FIG_N - 1) // (N_VAL - 1)                    # PINN profile on the ref x-nodes
    H = Y_MAX - Y_MIN
    row = {"pe": pe, "ckpt": label, **errs,
           "max_abs_T": float(np.abs(T - T_ref).max()),
           "max_at": (float(val["X"][j, i]), float(Y[j, i])),
           "rT_rms": r_T.pow(2).mean().sqrt().item()}
    for w in ("bot", "top"):
        a, b = wall[f"q_{w}"][::step], wall_ref[f"q_{w}"]
        row[f"Nu_{w}"] = H * _trapz(wall[f"q_{w}"], 1.0) / (HEAT_FIG_N - 1)
        row[f"Nu_{w}_ref"] = H * _trapz(b, 1.0) / (N_VAL - 1)
        row[f"eps_q_{w}"] = np.linalg.norm(a - b) / np.linalg.norm(b)
    for src, d in (("", wall), ("_ref", wall_ref)):
        row[f"Q_in{src}"], row[f"Q_out{src}"], row[f"Q_wall{src}"] = d["Q_in"], d["Q_out"], d["Q_wall"]
        row[f"imbalance{src}"] = (d["Q_in"] - d["Q_out"] - d["Q_wall"]) / d["Q_in"]
    fields = {"T": T, "T_ref": T_ref, "X": val["X"].cpu().numpy(), "Y": Y,
              "wall": wall, "wall_ref": wall_ref, "history": history}
    return row, fields


def heat_summary(rows, path="heat_summary.md"):
    flow = {}
    f = os.path.join(CKPT_DIR, "conv_4x50_final.pt")
    if os.path.exists(f):
        flow = torch.load(f, map_location="cpu", weights_only=False)["info"]
    pct = lambda v: f"{100 * v:.3f}"
    lines = ["# Heat-transfer extension: results (4x50, joint u, v, p, T network)", "",
             "Errors are relative L2 (%) on the 201 x 201 grid against the exact flow and the "
             "FD reference T.", "",
             "| Pe | checkpoint | eps_u | eps_v | eps_p | eps_T | max abs T err (at x, y) "
             "| RMS r_T |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['pe']:g} | {r['ckpt']} | {pct(r['eps_u'])} | {pct(r['eps_v'])} | "
                     f"{pct(r['eps_p'])} | {pct(r['eps_T'])} | {r['max_abs_T']:.1e} "
                     f"({r['max_at'][0]:.2f}, {r['max_at'][1]:.2f}) | {r['rT_rms']:.1e} |")
    if flow:
        lines.append(f"| flow only | conv_4x50 | {pct(flow['eps_u'])} | {pct(flow['eps_v'])} | "
                     f"{pct(flow['eps_p'])} | - | - | - |")
    lines += ["", "## Wall heat transfer and energy balance", "",
              "Nu = H q / dT_ref (H = 2, dT_ref = 1), mean over the wall; eps_q = rel. L2 error "
              "of the local flux profile. Imbalance = (Q_in - Q_out - Q_wall) / Q_in.", "",
              "| Pe | Nu bot (PINN / ref) | Nu top (PINN / ref) | eps_q bot | eps_q top "
              "| Q_wall/Q_in (ref) | imbalance PINN | imbalance ref |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['pe']:g} | {r['Nu_bot']:.3f} / {r['Nu_bot_ref']:.3f} | "
                     f"{r['Nu_top']:.3f} / {r['Nu_top_ref']:.3f} | {pct(r['eps_q_bot'])}% | "
                     f"{pct(r['eps_q_top'])}% | {r['Q_wall_ref'] / r['Q_in_ref']:.3f} | "
                     f"{100 * r['imbalance']:+.3f}% | {100 * r['imbalance_ref']:+.4f}% |")
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _pe_colours(pes):
    """Sequential single hue, light -> dark with Pe (magnitude, not identity)."""
    import numpy as np
    cmap = plt.get_cmap("Blues")
    return {pe: cmap(t) for pe, t in zip(pes, np.linspace(0.4, 0.95, len(pes)))}


def fig_heat_fields(rows, fields):
    """Columns = Pe; rows FD reference T | PINN T | |error|."""
    import numpy as np
    pes = [r["pe"] for r in rows]
    from matplotlib.ticker import MaxNLocator
    fig, axes = plt.subplots(3, len(pes), figsize=(FULL_W, 5.4), constrained_layout=True,
                             squeeze=False)
    levels = np.linspace(0, 1, 21)
    letters = iter("abcdefghijklmnopqrstuvwxyz")
    for c, pe in enumerate(pes):
        F = fields[pe]
        for r_, (data, title) in enumerate(((F["T_ref"], "FD reference"), (F["T"], "PINN"))):
            cf = axes[r_, c].contourf(F["X"], F["Y"], data, levels=levels, cmap="inferno",
                                      extend="both", zorder=-1)
            if r_ == 0:
                axes[r_, c].set_title(f"Pe = {pe:g}")
        ce = axes[2, c].contourf(F["X"], F["Y"], np.abs(F["T"] - F["T_ref"]), levels=30,
                                 cmap="magma", zorder=-1)
        cb = fig.colorbar(ce, ax=axes[2, c], orientation="horizontal", shrink=0.9, aspect=15)
        cb.locator = MaxNLocator(3)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.tick_params(labelsize=6)
        cb.update_ticks()
        for r_ in range(3):
            _field_axes(axes[r_, c], xlabel=r_ == 2, ylabel=c == 0)
            _panel(axes[r_, c], next(letters))
    for r_, name in enumerate(("FD reference\n", "PINN\n", r"$|T_{\mathrm{PINN}} - T_{\mathrm{ref}}|$" + "\n")):
        axes[r_, 0].set_ylabel(name + "$y$")
    fig.colorbar(cf, ax=axes[:2, :], shrink=0.6, aspect=30, label="$T$")
    _save(fig, "fig9_heat_fields")


def fig_heat_errors(rows):
    """(a) errors vs Pe, flow-only level dashed; (b) wall-flux error and energy imbalance."""
    pes = [r["pe"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.8), constrained_layout=True)
    ax = axes[0]
    flow = {}
    f = os.path.join(CKPT_DIR, "conv_4x50_final.pt")
    if os.path.exists(f):
        flow = torch.load(f, map_location="cpu", weights_only=False)["info"]
    styles = {"T": ("0.15", "s"), "u": (SERIES["u"], "o"), "v": (SERIES["v"], "^"),
              "p": (SERIES["p"], "D")}
    for var, (col, m) in styles.items():
        ax.plot(pes, [100 * r[f"eps_{var}"] for r in rows], color=col, marker=m, ms=5,
                lw=1.4 if var == "T" else 1.0, label=f"$\\varepsilon_{var}$")
        if var in "uvp" and flow:
            ax.axhline(100 * flow[f"eps_{var}"], color=col, ls=":", lw=0.9)
    ax.set(xscale="log", yscale="log", xlabel="Pe", ylabel=r"relative $L_2$ error (%)")
    ax.set_title("joint (u, v, p, T) network", fontsize=8.5)
    ax.plot([], [], color="0.4", ls=":", lw=0.9, label="flow-only\n4$\\times$50")
    ax.legend(ncol=2, fontsize=7)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    _panel(ax, "a")
    ax = axes[1]
    ax.plot(pes, [100 * r["eps_q_bot"] for r in rows], color="0.15", marker="v", ms=5,
            label="wall flux error, bottom")
    ax.plot(pes, [100 * r["eps_q_top"] for r in rows], color="0.15", marker="^", ms=5,
            ls="--", label="wall flux error, top")
    ax.plot(pes, [100 * abs(r["imbalance"]) for r in rows], color=SERIES["v"], marker="o",
            ms=5, label="|energy imbalance| (PINN)")
    ax.set(xscale="log", yscale="log", xlabel="Pe", ylabel="%")
    ax.set_title("derived heat-transfer quantities", fontsize=8.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    _panel(ax, "b")
    _save(fig, "fig10_heat_errors")


def fig_heat_nusselt(rows, fields):
    """(a, b) local Nu(x) on bottom / top wall, PINN solid vs reference dashed;
    (c) wall-mean Nu vs Pe."""
    pes = [r["pe"] for r in rows]
    col = _pe_colours(pes)
    H = Y_MAX - Y_MIN
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 3.0), layout="constrained")
    for k, w in enumerate(("bot", "top")):
        ax = axes[k]
        for pe in pes:
            F = fields[pe]
            ax.plot(F["wall"]["x"], H * F["wall"][f"q_{w}"], color=col[pe], lw=1.2,
                    label=f"Pe = {pe:g}")
            ax.plot(F["wall_ref"]["x"], H * F["wall_ref"][f"q_{w}"], color="0.1", lw=0.7,
                    ls=(0, (2, 2)))
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set(xlabel="$x$", ylabel="local Nu" if k == 0 else None)
        ax.set_title(f"{'bottom' if w == 'bot' else 'top'} wall $y = {Y_MIN if w == 'bot' else Y_MAX:g}$",
                     fontsize=8.5)
        ax.grid(alpha=0.25, lw=0.5)
        _panel(ax, "ab"[k])
    axes[0].plot([], [], color="0.1", lw=0.7, ls=(0, (2, 2)), label="FD reference")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=6,
               fontsize=7, title="PINN (solid)", title_fontsize=7)
    ax = axes[2]
    for w, m, ls in (("bot", "v", "-"), ("top", "^", "--")):
        ax.plot(pes, [r[f"Nu_{w}"] for r in rows], color="0.15", marker=m, ms=5, ls=ls,
                label=f"PINN, {'bottom' if w == 'bot' else 'top'}")
        ax.plot(pes, [r[f"Nu_{w}_ref"] for r in rows], ls="none", marker="o", ms=9,
                mfc="none", mec=SERIES["v"], mew=0.9)
    ax.plot([], [], ls="none", marker="o", ms=9, mfc="none", mec=SERIES["v"], label="FD reference")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set(xscale="log", xlabel="Pe", ylabel="wall-mean Nu")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, lw=0.5, which="both")
    _panel(ax, "c")
    _save(fig, "fig11_heat_nusselt")


def run_heat_analysis(pe_list=PE_LIST):
    _paper_style()
    rows, fields = [], {}
    for pe in pe_list:
        row, F = heat_analysis_one(pe)
        if row is None:
            print(f"Pe {pe:g}: no checkpoint, skipped")
            continue
        rows.append(row)
        fields[pe] = F
    heat_summary(rows)
    fig_heat_fields(rows, fields)
    fig_heat_errors(rows)
    fig_heat_nusselt(rows, fields)


# %% Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["checks", "trial", "baseline", "validate",
                                         "stage7", "run", "summary", "figures", "figures7",
                                         "heat_ref", "heat_trial", "heat", "heat_analysis"])
    parser.add_argument("--exp", help="'run' mode: one name from EXPERIMENTS")
    parser.add_argument("--pe", type=float, default=28.0, help="heat modes: Peclet number")
    parser.add_argument("--only", nargs="+", help="'stage7' mode: run just these experiments")
    parser.add_argument("--workers", type=int, default=3,
                        help="'stage7' mode: experiments run at the same time")
    parser.add_argument("--resume", metavar="CKPT",
                        help="Adam checkpoint to resume from (runs L-BFGS only)")
    parser.add_argument("--ckpt", default=os.path.join(CKPT_DIR, "baseline_final.pt"),
                        help="checkpoint to evaluate in 'validate' mode")
    args = parser.parse_args()


    if args.mode == "checks":
        run_checks()
    elif args.mode == "trial":
        run_experiment("trial", adam_schedule=[(2_000, 1e-3)], lbfgs_iters=500,
                       resume_from=args.resume)
    elif args.mode == "baseline":
        run_experiment("baseline", adam_schedule=PAPER_ADAM, lbfgs_iters=50_000,
                       resume_from=args.resume)
    elif args.mode == "validate":
        run_validation(args.ckpt)
    elif args.mode == "run":
        run_named(args.exp)
    elif args.mode == "stage7":
        run_queue(args.only or list(EXPERIMENTS), args.workers)
    elif args.mode == "summary":
        stage7_summary()
    elif args.mode == "figures":
        DEVICE = torch.device("cpu")     # plotting only: never compete with training for GPU memory
        make_paper_figures(args.ckpt)
    elif args.mode == "heat_ref":
        heat_reference()
    elif args.mode == "heat_trial":
        run_heat(f"heat_trial_Pe{args.pe:g}", args.pe, adam_schedule=[(2_000, 1e-3)],
                 lbfgs_iters=500, plateau=None)
    elif args.mode == "heat":
        run_heat(f"heat_Pe{args.pe:g}", args.pe)
    elif args.mode == "heat_analysis":
        DEVICE = torch.device("cpu")     # evaluation only; GPU is busy with training
        torch.set_num_threads(8)
        run_heat_analysis()
    elif args.mode == "figures7":
        DEVICE = torch.device("cpu")
        make_stage7_figures()
