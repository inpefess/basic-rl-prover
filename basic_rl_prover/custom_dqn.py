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
from typing import Optional, Type

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.apex_dqn import ApexDQN
from ray.rllib.policy import Policy
from ray.rllib.utils.annotations import override

from basic_rl_prover.custom_dqn_torch_policy import CustomDQNTorchPolicy


# pylint: disable=abstract-method
class CustomDQN(ApexDQN):
    """DQN patched with policy including trajectory post-processing."""

    @classmethod
    @override(ApexDQN)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        """Return a default Policy class to use, given a config."""
        if config["framework"] == "torch":
            return CustomDQNTorchPolicy
        raise NotImplementedError("don't work with anything but torch")
