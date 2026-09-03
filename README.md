# Safe Autonomous Highway Driving — DQN vs Double DQN

**Option DDQN-1** · MSc Data Science, Reinforcement Learning
Environment: `highway-v0` (highway-env) · Gymnasium API · Stable-Baselines3

---

## What this project is actually about

The brief lets you take the algorithm from a library. So the project is **not**
"implement Double DQN". Section 3 says the assessed work is:

| Assessed contribution | Where it lives here |
|---|---|
| MDP formulation and state justification | `docs/01_mdp_formulation.md` |
| Environment wrapper, observation, termination | `envs/` |
| Reward function and its weighting | `envs/reward.py` |
| Baseline implementation | `configs/dqn.yaml` + stock SB3 |
| Evaluation harness, seeding, aggregation | `evaluation/` |
| Logging, plotting, analysis | `agents/callbacks.py`, `scripts/plot.py` |

The algorithm itself is ~10 lines in `agents/double_dqn.py`. Everything else is
the actual submission.

---

## Quick start

```bash
# 1. Install (see the version trap noted in requirements.txt)
pip install -r requirements.txt

# 2. Verify everything works -- takes about 3 minutes
python tests/test_pipeline.py

# 3. Smoke test the full pipeline -- about 5 minutes, results are throwaway
python scripts/run_all.py --smoke

# 4. The real experiment -- about 6 hours on CPU
python scripts/run_all.py
```

Or run the stages individually:

```bash
python scripts/train.py --config configs/dqn.yaml     # baseline, 3 seeds
python scripts/train.py --config configs/ddqn.yaml    # treatment, 3 seeds
python scripts/evaluate.py --include-random           # metrics + statistics
python scripts/plot.py                                # all five figures
```

---

## How long it takes, and why

Measured on CPU in this project's own configuration:

| `vehicles_count` | env steps/sec | One 60k-step run | Full 2×3-seed experiment |
|---|---|---|---|
| 50 (library default) | 3.7 | 7.5 hours | **45 hours** |
| 20 (our setting) | 16.2 | 1.0 hour | **6 hours** |

The simulator advances every vehicle's IDM model 15 times per agent step, so cost
is linear in vehicle count. We use `vehicles_count: 20` with
`vehicles_density: 1.3` — fewer cars overall, packed closer together, so the
traffic the ego actually interacts with stays dense while the experiment becomes
feasible. This is a deliberate, measured trade-off and it is documented in
`configs/base.yaml` rather than hidden.

If you have more time, raise `total_timesteps` before raising `vehicles_count`.

---

## Repository layout

```
code/
├── docs/
│   └── 01_mdp_formulation.md    THE formulation -- read this first
├── envs/                        ← assessed original work
│   ├── observation.py           state construction (Section 1)
│   ├── reward.py                our reward function (Section 3)
│   ├── monitoring.py            the five required metrics
│   └── make_env.py              the single place an env is built
├── agents/
│   ├── double_dqn.py            the ~10-line difference from the baseline
│   └── callbacks.py             Q-value probe + per-episode logging
├── evaluation/                  ← assessed original work
│   ├── harness.py               seeding, metrics, aggregation
│   └── stats.py                 paired bootstrap, effect sizes
├── configs/
│   ├── base.yaml                everything shared -- annotated with SB3 defaults
│   ├── dqn.yaml                 baseline   (differs only in `name`)
│   └── ddqn.yaml                treatment  (differs only in `name`)
├── scripts/
│   ├── train.py    evaluate.py    plot.py    run_all.py
├── tests/
│   └── test_pipeline.py         37 checks, including the critical target test
└── results/                     produced by the scripts
```

---

## The formulation in one table

Full reasoning in `docs/01_mdp_formulation.md`.

| Element | Specification |
|---|---|
| State | ℝ⁹⁶ — 4 stacked frames × 24 features, normalised to [−1,1] |
| Observation | `Kinematics`, 5 vehicles × 5 features, ego-relative, ego absolute *x* dropped |
| Actions | Discrete(5): lane left/right, faster, slower, idle |
| Reward | 6 weighted terms: speed, time-headway, lane-change cost, invalid action, collision, off-road |
| Termination | collision ∨ off-road (absorbing — no bootstrap) |
| Truncation | t ≥ 40 steps (**not** absorbing — must bootstrap) |
| γ | 0.95, giving a 20-step ≈ 20-second effective horizon |
| Markov | Violated per-frame (no acceleration); compensated by 4-frame stacking |

---

## What makes the comparison valid

These are the things a marker will look for.

**One difference only.** `configs/dqn.yaml` and `configs/ddqn.yaml` both inherit
`base.yaml` and change nothing but the algorithm name. `DoubleDQN` subclasses SB3's
`DQN` and overrides `train()` alone. Any performance gap can only come from the
target computation.

**Three seeds, held constant.** Seeds `[0, 1, 2]`, recorded in every
`manifest.json`. Identical hyperparameters across seeds, as the brief requires.

**Evaluation seeds disjoint from training seeds.** Training uses 0–2; evaluation
uses 1000–1049. Testing on the training set is a real and common mistake.

**Paired evaluation.** Evaluation episode *i* uses seed `1000 + i` for *every*
policy, so DQN-seed-0 and DDQN-seed-2 face byte-identical traffic. This removes
traffic luck as a confounder and sharply tightens the comparison.

**Aggregation over seeds, not episodes.** `aggregate_across_seeds` reports mean ±
std over the three *training runs*. Pooling 150 episodes as if they were 150
independent samples would understate the uncertainty — episodes within one run
share a network and are not independent.

**A performance floor.** `--include-random` evaluates a uniform-random policy. On
this task it crashes in essentially every episode, which is what makes "collision
rate 0.34" interpretable.

---

## Be prepared for a null result

Double DQN may not beat DQN here. Overestimation bias matters most when
action-values are noisy and the action set is large; this task has five actions
and a fairly dense reward, so the effect may sit inside seed noise.

**That is not a failed project.** The brief rewards rigour, not a positive result.
This is exactly why `agents/callbacks.py` probes Q-values on a fixed set of states
throughout training: it lets you demonstrate the *mechanism* Double DQN targets
even when the performance difference washes out.

`tests/test_pipeline.py` already proves the mechanism analytically — on drifted
networks the DQN target is systematically **above** the Double DQN target and
never below. A report that shows reduced Q-value inflation plus an honest
"performance difference within one standard deviation across three seeds" is
stronger than an unreplicated win.

---

## A subtlety worth knowing

When the online and target networks are identical — which is true at
initialisation and immediately after every hard target sync — **DQN and Double
DQN compute exactly the same target**. If `Q_online == Q_target`, then
`argmax_a Q_online(s',a)` is precisely the action attaining `max_a Q_target(s',a)`.

The algorithms only diverge as the online network drifts between syncs. This means
`target_update_interval` directly controls how much opportunity Double DQN has to
matter: sync every step and the two are identical. Ours is 500 steps. That is a
genuinely interesting thing to discuss in the report, and it is verified in
`tests/test_pipeline.py`.

---

## Outputs

After a full run, `results/analysis/` contains:

| File | Contents |
|---|---|
| `evaluation_episodes.csv` | one row per evaluation episode |
| `summary_per_seed.csv` | one row per (algorithm, seed) |
| `summary_aggregate.csv` | mean ± std across seeds — the headline numbers |
| `comparison.csv` / `.md` | the DQN-vs-DDQN statistical comparison |
| `figures/01_training_return.png` | return vs steps, mean across seeds + spread |
| `figures/02_collision_rate.png` | collision rate during training |
| `figures/03_q_values.png` | probed Q-values — the overestimation evidence |
| `figures/04_eval_comparison.png` | final metrics, error bars across seeds |
| `figures/05_reward_components.png` | which reward terms each agent earns |

Plus, per run, `results/<algo>/hyperparameters.md` — the hyperparameter table the
brief requires, generated from the config so it cannot drift from the code.
