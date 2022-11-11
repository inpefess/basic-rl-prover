#   Copyright 2022 Boris Shminke
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
# noqa: D205, D400
"""
DQN with customised policy
==========================
"""
import sys
from typing import Optional, Type

from ray.rllib.algorithms.dqn import DQN
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ResultDict

from basic_rl_prover.custom_dqn_torch_policy import CustomDQNTorchPolicy


# pylint: disable=abstract-method
class CustomDQN(DQN):
    """DQN patched with policy including trajectory post-processing."""

    @override(DQN)
    def get_default_policy_class(self, config: dict) -> Optional[Type[Policy]]:
        """Return a default Policy class to use, given a config."""
        if config["framework"] == "torch":
            return CustomDQNTorchPolicy
        raise NotImplementedError("don't work with anything but torch")

    @override(DQN)
    def training_step(self) -> ResultDict:
        """Don't train if the buffer is empty.

        :returns: the results dict from executing the training iteration.
        """
        if not self.local_replay_buffer or len(self.local_replay_buffer) == 0:
            self.config[
                "num_steps_sampled_before_learning_starts"
            ] = sys.maxsize
        else:
            self.config["num_steps_sampled_before_learning_starts"] = 1
        return super().training_step()
