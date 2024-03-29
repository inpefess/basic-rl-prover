#   Copyright 2022-2023 Boris Shminke
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
Policy with trajectory post-processing
======================================
"""
from typing import Dict, Optional, Tuple

from gym_saturation.utils import get_positive_actions
from ray.rllib.algorithms.dqn import DQNTorchPolicy
from ray.rllib.evaluation import Episode
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID


def spread_reward(sample_batch: SampleBatch) -> None:
    """
    Propagates reward from the last step of a proof to all relevant steps.

    :param sample_batch: batch is modified by this function!
    :returns:
    """
    positive_actions = get_positive_actions(
        sample_batch[SampleBatch.INFOS][-1]["real_obs"]
    )
    proof_length = len(
        set(positive_actions).intersection(
            set(sample_batch[SampleBatch.ACTIONS])
        )
    )
    new_rewards = sample_batch[SampleBatch.REWARDS].copy()
    for i, action in enumerate(sample_batch[SampleBatch.ACTIONS]):
        if action in positive_actions:
            new_rewards[i] = 1.0 / proof_length
    sample_batch[SampleBatch.REWARDS] = new_rewards


# pylint: disable=abstract-method
class CustomDQNTorchPolicy(DQNTorchPolicy):
    """
    Custom policy based on standard DQN one.

    >>> from gymnasium.spaces import Discrete
    >>> dummy_policy = CustomDQNTorchPolicy(Discrete(2), Discrete(2), {})
    >>> sample_batch = getfixture("sample_batch") # noqa: F821
    >>> dummy_policy.postprocess_trajectory(sample_batch[2:])[
    ...     SampleBatch.REWARDS
    ... ]
    array([0.5, 0.5])
    """

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple[Policy, SampleBatch]]
        ] = None,
        episode: Optional[Episode] = None,
    ) -> SampleBatch:
        """Spread reward and generate new problems.

        :param sample_batch: batch of experiences for the policy, which will
            contain at most one episode trajectory.
        :param other_agent_batches: In a multi-agent env, this contains a
            mapping of agent ids to (policy, agent_batch) tuples containing the
            policy and experiences of the other agents.
        :param episode: An optional multi-agent episode object to provide
            access to all of the internal episode state, which may be useful
            for model-based or multi-agent algorithms.
        :returns: the post-processed sample batch.
        """
        if sample_batch[SampleBatch.REWARDS].max() > 0:
            spread_reward(sample_batch)
        return super().postprocess_trajectory(sample_batch)
