# Physics-Informed Neural Networks for Kovasznay Flow and Heat Transfer

PyTorch reimplementation of the velocity–pressure NSFnet (**VP-NSFnet**) of
[Jin, Cai, Li & Karniadakis (2021)](https://arxiv.org/abs/2003.06496) for steady 2D incompressible
Navier–Stokes flow (Kovasznay flow, Re = 40), a 21-run parametric study, and an **extension to passive
heat transfer** (Péclet number 1–280) validated against a finite-difference solver.

*Course project, CHE657 (Machine Learning in Chemical Engineering), IIT Kanpur, Oct–Nov 2026.
Instructor: Prof. Salman Ahmed Khan.*

## Key results

Relative L2 errors (%) on a 201 × 201 grid for the same 4 × 50 network:

| | ε_u | ε_v | ε_p |
|---|---|---|---|
| Paper, Table 1 (best of 10 runs) | 0.076 | 0.412 | 0.516 |
| **This work** (Adam + float64 L-BFGS) | **0.010** | **0.092** | **0.044** |
| This work, converged L-BFGS (loss-rescaling fix) | 0.004 | 0.037 | 0.014 |

- **The optimiser matters most.** Adam alone is 30–95× worse; boundary weight α (10–1000) and the number of
  training points matter little once L-BFGS has converged.
- **Network size barely matters** once converged: every size from 4 × 50 to 10 × 200 reaches ε_u ≈ 0.003–0.004 %.
- **Seed spread** (5 seeds): ε_v = 0.077 ± 0.011 %.
- **Heat extension:** one network learns (u, v, p, T). ε_T < 0.1 % and wall Nusselt number within 1.5 % of the
  finite-difference reference for Pe ≤ 100. At Pe = 280 (water) thin thermal layers appear: ε_T = 3.0 %, the
  bottom-wall Nusselt number is 14 % low, and the flow accuracy of the shared network degrades 12–19×.
  Energy is conserved within 0.07 % at every Pe, although the integral law is never in the loss.

Full results, derivations and discussion are in [`report/term_project_report.pdf`](report/term_project_report.pdf).

## Two PyTorch L-BFGS pitfalls found (and fixed)

1. **float32 line-search failure:** near the minimum the loss changes by less than float32 can resolve, so the
   strong-Wolfe line search returns a zero step. Fix: cast the model and data to float64 before L-BFGS.
2. **Hard-coded curvature threshold:** PyTorch stores a curvature pair only if `y·s > 1e-10`, which silently
   freezes L-BFGS when the loss is around 1e-5. Fix: minimise `L / L0` (same minimiser), with the gradient
   tolerance rescaled.

Both bugs looked like physics ("bigger networks are worse") until the logs were inspected.

## Installation

```bash
git clone https://github.com/Mayank020923/PINN-Kovasznay-Flow-Heat-Transfer.git
cd PINN-Kovasznay-Flow-Heat-Transfer
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python nsfnets_kovasznay.py checks                      # finishes in seconds
python nsfnets_kovasznay.py trial                       # ~1 min on a GPU, a few min on a CPU
```

A GPU is used automatically if PyTorch can see one, otherwise the CPU. Run all commands from the project
folder (the code uses the relative paths `figures/`, `checkpoints/`, `logs/`, `heat/`).

## Usage

Everything is one file, `nsfnets_kovasznay.py` (about 1,900 lines), driven by one command-line mode:

| Command | What it does | Typical time |
|---|---|---|
| `python nsfnets_kovasznay.py checks` | Sanity checks of the exact solution, sampling and gradients | seconds |
| `... trial` | Short run (2k Adam + 500 L-BFGS) to check the pipeline | ~1 min (GPU) |
| `... baseline` | Paper recipe: 110k Adam + L-BFGS, 4 × 50 network | ~1.2 h (GPU) |
| `... baseline --resume CKPT` | Skip Adam, load a checkpoint, run only L-BFGS | ~35 min |
| `... validate --ckpt CKPT` | Error report, field and residual plots for a checkpoint | seconds |
| `... figures --ckpt CKPT` | Publication figures fig1–fig4 | < 1 min |
| `... stage7 --workers N` | Parametric study (21 runs), N parallel processes, restartable | ~20 h with 4 workers |
| `... run --exp NAME` | One experiment from the `EXPERIMENTS` table | 0.1–18 h |
| `... summary` | Stage 7 result tables → `stage7_summary.md` | seconds |
| `... figures7` | Stage 7 figures (fig5–fig8) | < 1 min |
| `... heat_ref` | Finite-difference reference temperature for each Pe | minutes |
| `... heat_trial --pe 28` | Short heat run | ~3 min (CPU) |
| `... heat --pe 28` | Full heat run (paper Adam schedule + L-BFGS with plateau stop) | ~4–6 h (CPU) |
| `... heat_analysis` | ε_T vs Pe, Nusselt numbers, energy balance, fig9–fig11 | ~5 min per new Pe |

Times were measured on an NVIDIA RTX A5000 (shared with another user) and are indicative only.

**Order matters** — some modes need files written by earlier ones:

- `validate` and `figures` use `checkpoints/baseline_final.pt` by default → run `baseline` first
  (or use the checkpoint included in this repo).
- `heat`, `heat_trial` and `heat_analysis` need `heat/ref_Pe*.pt` → run `heat_ref` first.
- Stage 7 runs `ref` and `conv_*` resume from Adam checkpoints, so run the Adam stages first.

To force a run onto the CPU: `CUDA_VISIBLE_DEVICES= python nsfnets_kovasznay.py heat --pe 28`.

## Method in brief

- **Problem:** Kovasznay flow, Re = 40, on x ∈ [−0.5, 1.0], y ∈ [−0.5, 1.5]. The exact solution is known, so
  every error is measured against the truth. Pressure is only defined up to a constant, so the mean-shifted
  pressure is compared.
- **Network:** fully connected, 4 × 50 tanh, inputs rescaled to [−1, 1], Xavier initialisation. Output (u, v, p),
  plus T for the heat extension.
- **Loss:** `L = L_e + α L_b` — mean squared Navier–Stokes residuals at 2601 random interior points (by automatic
  differentiation) plus a boundary mismatch on 400 points, α = 100. Heat adds the energy residual
  `u T_x + v T_y − (T_xx + T_yy)/Pe` and its boundary term.
- **Training:** Adam (paper schedule, 110k steps, float32), then L-BFGS (float64, rescaled loss).
- **Heat model:** passive scalar carried by the Kovasznay flow, `Pe = Re · Pr` (air 28, water 280). The reference
  is a second-order finite-difference solution on 401/801/1601 grids with observed order 2.00.

## Repository structure

```
nsfnets_kovasznay.py        all code (stages 1–7 and heat extension H1–H5)
requirements.txt
README.md
report/
  term_project_report.pdf   full project report
docs/
  code_guide.pdf            line-by-line guide to the code
figures/                    result figures (fields, training history, Nusselt number, ...)
results/
  stage7_summary.md         parametric-study tables
  heat_summary.md           heat-transfer tables
checkpoints/
  baseline_final.pt         trained baseline 4 × 50 model
presentation/               slides (optional)
```

Large generated folders (`logs/`, `heat/`, most `checkpoints/`) are not tracked; see `.gitignore`.
Re-create them with the commands above.

## Notes

- Do not run L-BFGS in float32 and do not remove the loss scaling in `train_lbfgs`; both make L-BFGS stop or
  freeze silently.
- Never start two jobs with the same experiment tag; they write the same checkpoint and log files.
- Checkpoints are loaded with `weights_only=False` because they contain Python lists and dicts. Only load
  checkpoint files you trust.
- The baseline numbers and the capped Stage 7 runs were produced before the loss-rescaling fix, so the current
  `baseline` mode converges further (the equivalent run `conv_4x50` gives 0.004 / 0.037 / 0.014 %).

## Reference

X. Jin, S. Cai, H. Li, G. E. Karniadakis, *NSFnets (Navier–Stokes flow nets): Physics-informed neural networks
for the incompressible Navier–Stokes equations*, J. Comput. Phys. 426 (2021) 109951. arXiv:2003.06496.

## Author

Mayank Kumar — Department of Chemical Engineering, IIT Kanpur.

## License

MIT (see `LICENSE`).
