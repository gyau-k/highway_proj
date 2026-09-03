"""
Double DQN, implemented as a minimal subclass of Stable-Baselines3' DQN.

THE WHOLE POINT OF THIS FILE
----------------------------
The baseline (`DQN`) and the treatment (`DoubleDQN`) must differ in EXACTLY ONE
THING: how the bootstrap target is computed. If they differed in anything else --
network size, buffer size, exploration schedule -- then any performance gap could
be attributed to that other difference, and the experiment would prove nothing.

So this file subclasses SB3's `DQN` and overrides `train()` only. Everything else
-- the replay buffer, the target-network sync, the epsilon schedule, the
optimiser -- is inherited unchanged from the library.


WHAT DOUBLE DQN ACTUALLY CHANGES (for beginners)
------------------------------------------------
Q-learning updates toward a target built from the best action in the next state.
Vanilla DQN computes that target as:

    y = r + gamma * max_a' Q_target(s', a')

The `max` is the problem. Q_target is a noisy estimate. Taking the maximum over
several noisy estimates systematically picks out whichever one happened to be
over-estimated by noise. Averaged over many updates this biases Q upward --
this is "maximisation bias" or "overestimation bias". The agent becomes
optimistic about actions it has simply been lucky with.

Double DQN splits the single `max` into two steps performed by two different
networks:

    a*  = argmax_a' Q_online(s', a')     <- the ONLINE net SELECTS the action
    y   = r + gamma * Q_target(s', a*)   <- the TARGET net EVALUATES it

Because the noise in the online net and the noise in the target net are not
perfectly correlated, an action that looks good to the online net purely by
chance is unlikely to look equally good to the target net. The bias shrinks.

An analogy: vanilla DQN is one person picking the best restaurant AND rating it.
Whatever they happened to overrate, they choose. Double DQN has one person pick
and a second person rate. The second person is not in on the first one's error.

Reference: van Hasselt, Guez & Silver (2016), "Deep Reinforcement Learning with
Double Q-learning", AAAI.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3 import DQN
from stable_baselines3.common.utils import polyak_update


class DoubleDQN(DQN):
    """DQN with the Double Q-learning target.

    Identical to `stable_baselines3.DQN` in every respect except the four lines
    marked `DOUBLE DQN` in `train()` below.
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (affects batch norm / dropout, if any).
        self.policy.set_training_mode(True)
        # Update the learning rate according to any schedule.
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        mean_target_q: list[float] = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )

            with th.no_grad():
                # ---------------------------------------------------------
                # DOUBLE DQN -- the entire difference from vanilla DQN.
                #
                # Vanilla DQN would do:
                #     next_q_values = self.q_net_target(next_obs).max(dim=1)
                #
                # We instead SELECT with the online net and EVALUATE with the
                # target net.
                # ---------------------------------------------------------
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_values = self.q_net_target(
                    replay_data.next_observations
                ).gather(1, next_actions)
                # ---------------------------------------------------------

                next_q_values = next_q_values.reshape(-1, 1)

                # `replay_data.dones` already excludes timeouts, because SB3's
                # ReplayBuffer is created with handle_timeout_termination=True.
                # That is what makes us bootstrap correctly through TRUNCATION
                # while not bootstrapping through TERMINATION -- exactly the
                # distinction drawn in formulation Section 4.
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.gamma * next_q_values
                )
                mean_target_q.append(target_q_values.mean().item())

            # Q-values for the actions actually taken.
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            )

            # Huber loss -- less sensitive to outliers than MSE.
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))
        self.logger.record("train/mean_target_q", float(np.mean(mean_target_q)))


# Registry so scripts can select an algorithm by name from a config file.
ALGORITHMS: dict[str, type[DQN]] = {
    "dqn": DQN,            # baseline -- stock Stable-Baselines3, unmodified
    "ddqn": DoubleDQN,     # treatment -- the subclass above
}


def get_algorithm(name: str) -> type[DQN]:
    key = name.strip().lower()
    if key not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm '{name}'. Choose from {sorted(ALGORITHMS)}.")
    return ALGORITHMS[key]
