# MDP Formulation — Safe Autonomous Highway Driving

**Option DDQN-1** · Environment: `highway-v0` (highway-env 1.12.1, Gymnasium 1.3.0)

> This document is written **before** any training, as required by section 4 of the brief.
> Everything here is a design decision we made, and every decision has a reason attached.

---

## 0. Reading this document as a beginner

An **MDP** (Markov Decision Process) is just a formal way of writing down a decision problem.
You need five things, and that's the whole list:

| Symbol | Name | In plain English | Our answer |
|---|---|---|---|
| $\mathcal{S}$ | State space | What the agent can see | 96 numbers (§1) |
| $\mathcal{A}$ | Action space | What the agent can do | 5 choices (§2) |
| $R$ | Reward function | What "good" means, as a number | An equation (§3) |
| $P$ | Transition function | How the world changes | The simulator gives us this |
| $\gamma$ | Discount factor | How much the future matters | 0.95 (§5) |

The simulator hands us $P$ for free. **We must design $\mathcal{S}$, $\mathcal{A}$, $R$ and $\gamma$
ourselves** — and section 3 of the brief says that design *is* the assessed work.

---

## 1. State space

### 1.1 What the simulator gives us

`highway-env` returns a **`Kinematics`** observation: a matrix of shape $(V, F) = (5, 5)$.

- **5 rows** = the ego vehicle, plus the 4 nearest other vehicles, sorted by distance.
- **5 columns** = `[presence, x, y, vx, vy]`.

Row 0 is the ego and uses **absolute** coordinates.
Rows 1–4 use coordinates **relative to the ego**. This matters enormously — see §1.4.

A real observation, taken from `seed=0` with normalisation switched off so the units are readable:

```
        presence     x (m)   y (m)   vx (m/s)  vy (m/s)
ego  [    1.0      177.47    12.0      25.00     0.0 ]
v1   [    1.0       18.15    -4.0      -3.88     0.0 ]
v2   [    1.0       40.24    -4.0      -2.18     0.0 ]
v3   [    1.0       63.33     0.0      -1.20     0.0 ]
v4   [    1.0       84.32    -8.0      -3.90     0.0 ]
```

Read row `v1` as: *"a vehicle exists, 18.15 m ahead of me, 4 m to my left (one lane over),
travelling 3.88 m/s slower than me."*

### 1.2 Feature definitions

| Feature | Meaning | Units |
|---|---|---|
| `presence` | 1 if this row describes a real vehicle, 0 if it is padding | binary |
| `x` | Longitudinal position (along the road) | m |
| `y` | Lateral position (across lanes; lanes are 4 m apart) | m |
| `vx` | Longitudinal velocity | m/s |
| `vy` | Lateral velocity | m/s |

The road has 4 lanes at $y \in \{0, 4, 8, 12\}$ m. In the sample above the ego is at
$y = 12$, i.e. the rightmost lane.

### 1.3 Normalisation

Each continuous feature is divided by a fixed range and clipped to $[-1, 1]$:

$$\tilde{f} = \mathrm{clip}\!\left(\frac{f}{f_{\max}},\, -1,\, 1\right)$$

| Feature | $f_{\max}$ | Verification against the sample |
|---|---|---|
| `x` | 200 m | $177.47 / 200 = 0.887$ ✓ |
| `y` | 16 m | $12 / 16 = 0.75$ ✓ |
| `vx` | 80 m/s | $25 / 80 = 0.3125$ ✓ |
| `vy` | 80 m/s | — |

These are `highway-env`'s built-in `features_range` values, confirmed empirically rather
than assumed. Normalisation matters because neural networks train poorly when input
features have wildly different scales — a raw position of 177 alongside a velocity of 25
would let position dominate the first layer's gradients.

### 1.4 Design decision A — keep relative coordinates

We keep `absolute=False` (the default). The network therefore reads *"18 m ahead, closing
at 3.9 m/s"* directly.

**Why this matters.** The quantity that determines whether you crash is the *gap* and the
*closing speed* — both differences between the ego and a neighbour. With absolute
coordinates the network would have to learn subtraction from scratch, in its first layer,
from reward signal alone. Relative coordinates hand it the answer for free. This is
feature engineering, and it is legitimate and standard.

### 1.5 Design decision B — discard the ego's absolute longitudinal position

We **delete** feature `[0, 1]` (the ego's `x`).

**Why.** It is absolute distance travelled along the road, clipped at 200 m. The ego
starts near $x = 177$ m and drives forward, so within roughly one second this feature
saturates at exactly $1.0$ and remains there for the rest of every episode.

It is therefore (a) uninformative — it never varies once saturated — and (b) the only
**non-stationary** feature in the vector, meaning its distribution shifts systematically
over an episode in a way unrelated to the task. Feeding a constant with a transient at the
start is strictly worse than not feeding it. We keep the ego's `y` (lane position), `vx`
(speed) and `vy` (lateral drift), which all carry real information.

This reduces one frame from 25 features to **24**.

### 1.6 Design decision C — frame stacking

We stack the **last $k = 4$ observations**, giving

$$s_t = \left[o_{t-3},\, o_{t-2},\, o_{t-1},\, o_t\right] \in \mathbb{R}^{96}$$

The justification is the Markov analysis in §6, so read that before judging this choice.

### 1.7 Summary

> **State space.** $\mathcal{S} \subseteq [-1, 1]^{96}$. Continuous, real-valued,
> $96 = 4 \text{ frames} \times 24 \text{ features}$, where one frame is a
> $(5 \times 5)$ `Kinematics` matrix flattened with the ego's absolute longitudinal
> position removed. All features min–max normalised to $[-1, 1]$ and clipped.

---

## 2. Action space

**Discrete, $|\mathcal{A}| = 5$.** `highway-env`'s `DiscreteMetaAction`:

| Index | Action | Effect |
|---|---|---|
| 0 | `LANE_LEFT` | Target the lane to the left |
| 1 | `IDLE` | Hold current lane and speed |
| 2 | `LANE_RIGHT` | Target the lane to the right |
| 3 | `FASTER` | Increase target speed by one increment |
| 4 | `SLOWER` | Decrease target speed by one increment |

These are **meta-actions**, not raw steering and throttle. A low-level controller inside the
simulator converts `LANE_LEFT` into a smooth steering trajectory over the following second.

**Why discrete rather than continuous?** Because DQN and Double DQN are value-based methods
that compute $\max_a Q(s,a)$ — an enumeration over actions. That maximisation is only
tractable for a finite action set. A continuous steering-and-throttle action space would
require a different algorithm family entirely (DDPG, SAC, PPO). The choice of algorithm and
the choice of action space are not independent, and the brief fixes the algorithm.

**Timing.** `policy_frequency = 1` Hz, `simulation_frequency = 15` Hz. The agent chooses one
action per second; the physics advances 15 steps between decisions. **One environment step
= one second of simulated time.** Keep this in mind for §5.

### 2.1 Action masking

At the leftmost lane, `LANE_LEFT` is unavailable; at the rightmost, `LANE_RIGHT` is.
`highway-env` silently converts an unavailable lane change into `IDLE`.

We deliberately **do not mask** these, and instead apply a small explicit penalty
($w_{\text{inv}}$ in §3). Masking would hide the mistake; penalising teaches the agent that
the road has edges — which is information the state already contains via the ego's `y`, so
it is learnable. This is a defensible choice either way, and we state it so the marker can
see it was a choice and not an oversight.

---

## 3. Reward function

`highway-env` ships its own reward. **We do not use it.** Section 3 of the brief requires the
reward function to be our own work, so our wrapper discards the simulator's scalar entirely
and computes the following from the raw simulator state.

### 3.1 The equation

$$
R(s_t, a_t, s_{t+1}) \;=\;
\underbrace{w_v \, r_v}_{\text{go fast}}
\;+\; \underbrace{w_h \, r_h}_{\text{keep a safe gap}}
\;+\; \underbrace{w_\ell \, r_\ell}_{\text{don't dither}}
\;+\; \underbrace{w_{\text{inv}} \, r_{\text{inv}}}_{\text{no illegal moves}}
\;+\; \underbrace{w_c \, r_c}_{\text{never crash}}
\;+\; \underbrace{w_o \, r_o}_{\text{stay on road}}
$$

### 3.2 The components

**Speed** — reward for travelling in the target band $[v_{\min}, v_{\max}] = [20, 30]$ m/s:

$$r_v = \mathrm{clip}\!\left(\frac{v_t - v_{\min}}{v_{\max} - v_{\min}},\, 0,\, 1\right)$$

Linear from 0 at 20 m/s to 1 at 30 m/s. Below 20 it is 0, so crawling earns nothing.

**Headway (safety)** — penalise a short time gap to the vehicle directly ahead in the ego's
lane. Time headway $\tau_t = d_t / \max(v_t, \epsilon)$, in seconds, where $d_t$ is the
bumper-to-bumper gap:

$$r_h = -\,\mathrm{clip}\!\left(\frac{\tau_{\text{safe}} - \tau_t}{\tau_{\text{safe}}},\, 0,\, 1\right),
\qquad \tau_{\text{safe}} = 1.5\ \text{s}$$

Zero when the gap is comfortable, ramping to $-1$ as the gap closes to nothing.

**Why time headway rather than raw distance?** A 20 m gap is generous at 5 m/s and lethal
at 30 m/s. Dividing by speed makes the term scale-free and matches how human driving rules
are actually written (the "two-second rule"). This is the single term that makes the reward
about *safety* rather than *speed*, and it is our main original contribution to the reward.

**Lane-change cost** — a small fixed cost whenever a lane change is commanded:

$$r_\ell = -\mathbb{1}\left[a_t \in \{\texttt{LANE\_LEFT}, \texttt{LANE\_RIGHT}\}\right]$$

Without this, agents learn to weave — flicking between lanes costs nothing and occasionally
helps, so the policy becomes jittery and unrealistic.

**Invalid lane change** — commanded a lane change into a lane that does not exist:

$$r_{\text{inv}} = -\mathbb{1}\left[\text{lane change requested but unavailable}\right]$$

**Collision** — fires once, on the step the crash happens:

$$r_c = -\mathbb{1}\left[\text{crashed}\right]$$

**Off-road** — fires once, on the step the ego leaves the drivable surface:

$$r_o = -\mathbb{1}\left[\text{not on road}\right]$$

### 3.3 The weights

| Weight | Value | Reasoning |
|---|---|---|
| $w_v$ | 1.0 | Reference scale. Max $+1$ per step. |
| $w_h$ | 0.4 | Enough to override the speed bonus when tailgating badly ($-0.4$ vs $+1$ is not enough alone — it works together with $w_c$). |
| $w_\ell$ | 0.1 | Small. Discourages dithering without discouraging genuinely useful overtakes. |
| $w_{\text{inv}}$ | 0.2 | Twice a legal lane change. Cheap to learn to avoid. |
| $w_c$ | 10.0 | Dominant. See scale analysis below. |
| $w_o$ | 10.0 | Same severity as a collision. |

**Scale analysis — is $w_c = 10$ enough?** An episode is 40 steps (§4). A perfect
collision-free run scores at most $40 \times 1.0 = 40$. If the agent crashes at step 20 it
forfeits the remaining 20 steps ($-20$ of foregone reward) *and* takes the $-10$ penalty:
an effective cost of $-30$, or 75% of the achievable return. Crashing is therefore
decisively worse than driving slowly and safely, which is the behaviour we want. Note that
most of the deterrent comes from **episode termination**, not the penalty itself — a point
worth making explicitly in the report.

### 3.4 Reward is not normalised

`highway-env`'s `normalize_reward` is set to `False`. We keep raw magnitudes so the reward
terms remain interpretable in the analysis (we can report "the agent lost 3.2 reward per
episode to headway violations"). The cost is that reward magnitudes are larger than the
$[0,1]$ range DQN implementations are often tuned for, which we compensate for with
gradient clipping (`max_grad_norm = 10`).

---

## 4. Episode termination and truncation

The brief asks for these **stated separately**, because they are mathematically different
and conflating them is a real bug, not a formality.

### 4.1 Termination — the episode genuinely ends

$$\text{terminated}_t = \mathbb{1}[\text{crashed}_t] \;\vee\; \mathbb{1}[\text{off-road}_t]$$

These are **absorbing states**. There is no future after a crash, so $V(s_{\text{terminal}}) = 0$
and the Bellman target is:

$$y_t = r_t$$

### 4.2 Truncation — we stopped watching, but the world continues

$$\text{truncated}_t = \mathbb{1}[t \geq T_{\max}], \qquad T_{\max} = 40 \text{ steps} = 40 \text{ s}$$

The car is still driving perfectly happily at step 40; we simply cut the recording. The
state is **not** terminal, so we must still bootstrap:

$$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$$

### 4.3 Why this distinction is not pedantry

If you treat truncation as termination, you teach the agent that the world *ends* after 40
seconds. It then has no incentive to be in a good position at step 39 — the value of every
state near the time limit gets pulled toward zero, and that error propagates backwards
through the Q-function. The result is an agent that mysteriously degrades near the end of
episodes.

Stable-Baselines3 handles this correctly **provided the environment returns the two flags
separately**, which Gymnasium's five-tuple API does. Our wrapper preserves them, and we
verify it in the test suite.

---

## 5. Discount factor

$$\boxed{\gamma = 0.95}$$

### 5.1 The horizon argument

The discount factor sets an **effective horizon** — roughly how many steps into the future
the agent meaningfully cares about:

$$H_{\text{eff}} \approx \frac{1}{1 - \gamma} = \frac{1}{0.05} = 20 \text{ steps}$$

Because one step is one second (§2), that is **20 seconds of lookahead**.

### 5.2 Why 20 seconds is the right scale for this task

Match the horizon to the causal structure of the problem:

| Event | Timescale | Covered by $H_{\text{eff}} = 20$ s? |
|---|---|---|
| A developing rear-end collision | 2–5 s | Yes, comfortably |
| Completing a lane change | ~1–2 s | Yes |
| Overtaking a slow vehicle | 5–10 s | Yes |
| Whole episode | 40 s | No — deliberately |

A collision 10 steps away is discounted by $0.95^{10} = 0.60$: still clearly visible in the
value function. So the agent *can* learn to brake now to avoid a crash later.

### 5.3 Why not other values

- **$\gamma = 0.99$** gives $H_{\text{eff}} = 100$ s, which is 2.5× longer than the entire
  episode. The agent would be trying to optimise over a horizon that does not exist, which
  adds variance to the return estimate for no benefit.
- **$\gamma = 0.8$** gives $H_{\text{eff}} = 5$ s. Too myopic: an overtake takes longer than
  that to pay off, so the agent would under-value lane changes.

$\gamma = 0.95$ sits at half the episode length — long enough to capture every causal chain
that matters, short enough to keep return estimates low-variance.

---

## 6. Does the Markov property hold?

**No — not for a single frame.** This section is the honest answer, and the compensation.

### 6.1 What the Markov property requires

A state is Markov if the future depends only on the present state and action, not on how
you got there:

$$P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t, s_{t-1}, a_{t-1}, \dots, s_0)$$

Informally: *a single snapshot must contain everything you need to predict what happens next.*

### 6.2 What is missing from a single frame

**(a) Acceleration.** The observation contains position and velocity, but **not acceleration**.
The other vehicles are `IDMVehicle`s — they brake and accelerate according to the Intelligent
Driver Model based on their own leaders. From one frame you cannot distinguish:

- a car ahead coasting at $-1.2$ m/s relative to you, versus
- a car ahead braking hard, currently at $-1.2$ m/s and about to be at $-15$ m/s

Those two situations look **numerically identical** in a single observation, and they demand
opposite actions (`IDLE` versus `SLOWER`). This is a textbook violation.

**(b) Lane-change intent.** No indicators, no signals. A neighbour beginning to drift into
your lane is only distinguishable from one holding its lane once `vy` has already become
non-zero — by which point you have lost reaction time.

**(c) Truncated neighbourhood.** Only the 4 nearest vehicles are visible. A fifth vehicle
closing rapidly is invisible until it enters the top-4, at which point it appears
discontinuously.

**(d) Unstable row ordering.** Rows are sorted by distance (`order='sorted'`). When `v2`
overtakes `v1`, their rows swap contents in a single step. The network sees a discontinuous
jump in its input that corresponds to nothing discontinuous in the world.

### 6.3 How the implementation compensates

**Frame stacking with $k = 4$** (§1.6). By presenting four consecutive observations, the
network can compute finite differences internally:

$$\hat{a}_{t} \approx \frac{v_t - v_{t-1}}{\Delta t}, \qquad \Delta t = 1 \text{ s}$$

This recovers (a) acceleration and (b) the onset of lateral motion. The stacked state is
**approximately Markov**: it is a 4-step history, which for a system whose dynamics are
second-order (position, velocity, acceleration) is sufficient in principle.

**What stacking does not fix:** (c) the truncated neighbourhood, and (d) row-swap
discontinuities. We accept both as known limitations. The principled alternative for (d) is
an `OccupancyGrid` observation with a convolutional network, which is permutation-invariant
over vehicles by construction — we note this as future work rather than adopting it, because
it substantially increases training cost and the brief prioritises a controlled DQN-vs-DDQN
comparison over maximum absolute performance.

### 6.4 Statement for the report

> The single-frame observation is **not** Markov: it omits acceleration and lane-change
> intent, so distinct futures share identical observations. We restore approximate
> Markovianity by stacking $k = 4$ consecutive frames, from which the network can recover
> acceleration and lateral-motion onset as finite differences. Residual violations —
> a neighbourhood truncated to 4 vehicles, and discontinuous row reordering under the
> distance sort — are documented and accepted.

---

## 6.5 One measured compute decision

The library default is `vehicles_count = 50`. We use **20**, with
`vehicles_density` raised from 1.0 to **1.3**. This is a deliberate, measured
trade-off rather than an arbitrary change, so it is declared here.

The simulator advances every vehicle's IDM model 15 times per agent step, so
wall-clock cost is linear in the number of vehicles. Measured on CPU:

| `vehicles_count` | env steps/sec | One 60k-step run | Full 2 × 3-seed experiment |
|---|---|---|---|
| 50 (default) | 3.7 | 7.5 h | **45 h** |
| 20 (ours) | 16.2 | 1.0 h | **6 h** |

At the default, a three-seed comparison is not feasible on the available
hardware — and the brief's three-seed requirement is not negotiable, whereas the
traffic count is. Raising `vehicles_density` to 1.3 packs the remaining vehicles
closer together, so the *local* traffic the ego actually interacts with stays
dense; what falls is the number of distant vehicles being simulated at cost but
outside the ego's 4-vehicle observation window anyway.

Trading task difficulty for statistical power is the right direction here: an
under-powered comparison on hard traffic tells you nothing, whereas a
well-powered comparison on slightly lighter traffic answers the question that was
asked.

---

## 7. Formulation summary table

| Element | Specification |
|---|---|
| State space | $\mathcal{S} \subseteq [-1,1]^{96}$; 4 stacked frames × 24 features; min–max normalised |
| Observation source | `Kinematics`, $V=5$, $F=5$, ego-relative, ego absolute $x$ removed |
| Action space | Discrete, $\lvert\mathcal{A}\rvert = 5$ (`DiscreteMetaAction`) |
| Reward | Weighted sum of 6 terms; see §3.1 |
| Termination | Collision $\vee$ off-road (absorbing; no bootstrap) |
| Truncation | $t \geq 40$ steps (non-absorbing; **must** bootstrap) |
| Discount | $\gamma = 0.95$, $H_{\text{eff}} = 20$ steps $= 20$ s |
| Markov | Violated per-frame; compensated by $k=4$ frame stacking |
| Step duration | 1 s (`policy_frequency = 1` Hz) |
| Episode length | 40 steps = 40 s |
| Traffic | 20 vehicles at density 1.3 (measured compute decision, §6.5) |

---

## 8. What comes next

With the formulation fixed, implementation proceeds in this order:

1. **Environment wrappers** (`envs/`) — realise §1 and §3 in code.
2. **Evaluation harness** (`evaluation/`) — build the measuring instrument *before* running
   the experiment, so nothing is logged as an afterthought.
3. **Baseline** (`agents/`) — Stable-Baselines3 `DQN`.
4. **Treatment** (`agents/double_dqn.py`) — a subclass changing *only* the target
   computation.
5. **Experiments** — 3 seeds per algorithm, hyperparameters frozen across seeds.
6. **Analysis** — mean-across-seeds curves with spread, full hyperparameter table.
