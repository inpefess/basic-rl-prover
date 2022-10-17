#   Copyright 2022 Boris Shminke
#
#   This file is a derivative work based on the original work of
#   The Ray Team (https://github.com/ray-project/ray) distributed
#   under the Apache 2.0 license.
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
"""
Customised DQN Trainer
======================
"""
from typing import Optional, Type

from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.execution.replay_ops import StoreToReplayBuffer
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import SampleBatchType

from basic_rl_prover.custom_dqn_policy import CustomDQNPolicy


# pylint: disable=too-few-public-methods
class CustomStoreToReplayBuffer(StoreToReplayBuffer):
    """
    an adder to a replay buffer that skips episodes with no reward

    this function is excluded from coverage report because it's run by Ray Tune
    in a separate process that where ``pytest`` lives
    """

    def __call__(self, batch: SampleBatchType):  # pragma: no cover
        if isinstance(batch, SampleBatch):
            if batch["rewards"].max() > 0:
                return super().__call__(batch)
        return batch


# pylint: disable=abstract-method
class CustomDQNTrainer(DQN):
    """
    a DQN trainer with custom PyTorch policy and replay buffer

    >>> custom_trainer = CustomDQNTrainer(config={"framework": "wow"})
    Traceback (most recent call last):
     ...
    NotImplementedError
    """

    @override(DQN)
    def get_default_policy_class(self, config: dict) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return CustomDQNPolicy
        raise NotImplementedError
