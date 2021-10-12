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
"""
Customised DQN policy
=====================
"""

import logging
import sys
from typing import Dict, Optional, Tuple

from gym_saturation.envs.saturation_env import (
    POSITIVE_ACTIONS,
    PROBLEM_FILENAME,
    STATE_DIFF_UPDATED,
)
from ray.rllib.agents.dqn.dqn import DQNTorchPolicy
from ray.rllib.evaluation import Episode
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def propagate_reward(sample_batch: SampleBatch) -> None:
    """
    propagates reward from the last step of a proof to all relevant steps

    :param sample_batch: batch is modified by this function!
    :returns:
    """
    positive_actions = [
        info[POSITIVE_ACTIONS]
        for info in sample_batch["infos"]
        if POSITIVE_ACTIONS in info
    ][0]
    proof_length = len(positive_actions)
    for i, action in enumerate(sample_batch["actions"]):
        if action in positive_actions:
            sample_batch["rewards"][i] = 1.0 / proof_length
    for info in sample_batch["infos"]:
        info.pop(STATE_DIFF_UPDATED)
        if POSITIVE_ACTIONS in info:
            info.pop(POSITIVE_ACTIONS)


# pylint: disable=abstract-method
class CustomDQNPolicy(DQNTorchPolicy):
    """patch post processing"""

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple[Policy, SampleBatch]]
        ] = None,
        episode: Optional[Episode] = None,
    ) -> SampleBatch:
        """
        This will be called on each trajectory fragment computed during policy
        evaluation. Each fragment is guaranteed to be only from one episode.
        The given fragment may or may not contain the end of this episode,
        depending on the `batch_mode=truncate_episodes|complete_episodes`,
        `rollout_fragment_length`, and other settings.

        >>> import numpy as np
        >>> from gym.spaces import Box, Discrete
        >>> clause1 = {"class": "Clause", "processed": True, "literals": [], "label": "this_is_a_test_case", "birth_step": 1, "inference_parents": ["initial"], "inference_rule": "success"}
        >>> clause0 = {"class": "Clause", "processed": True, "literals": [{"class": "Literal", "negated": False, "atom": {"class": "Predicate", "name": "this_is_a_test_case", "arguments": []}, }], "label": "initial", "birth_step": 0, "inference_parents": None, "inference_rule": None}
        >>> policy = CustomDQNPolicy(
        ...     obs_space=Box(0, 1, (2, 2)), action_space=Discrete(2),
        ...     config={}
        ... )
        >>> trajectory = SampleBatch(
        ...     infos=[
        ...         {
        ...             STATE_DIFF_UPDATED: {0: clause0},
        ...             PROBLEM_FILENAME: "test"
        ...         },
        ...         {
        ...             STATE_DIFF_UPDATED: {1: clause1},
        ...             POSITIVE_ACTIONS: (0, 1),
        ...             PROBLEM_FILENAME: "test",
        ...         },
        ...     ],
        ...     rewards=np.array([0.0, 1.0]),
        ...     actions=np.array([0, 1]),
        ... )
        >>> print(policy.postprocess_trajectory(trajectory)["rewards"])
        [0.5 0.5]

        :param sample_batch: batch of experiences for the policy,
            which will contain at most one episode trajectory.
        :param other_agent_batches: In a multi-agent env, this contains a
            mapping of agent ids to (policy, agent_batch) tuples
            containing the policy and experiences of the other agents.
        :param episode: An optional multi-agent episode object to provide
            access to all of the internal episode state, which may
            be useful for model-based or multi-agent algorithms.
        :returns: The postprocessed sample batch.
        """
        # Do all post-processing always with no_grad().
        # Not using this here will introduce a memory leak
        # in torch (issue #6962).
        with self._no_grad_context():
            # Call super's postprocess_trajectory first.
            sample_batch = super().postprocess_trajectory(
                sample_batch, other_agent_batches, episode
            )
            if isinstance(sample_batch["infos"][0], dict):
                logger.info(
                    "EPISODE %s %d %d %s",
                    sample_batch["infos"][0][PROBLEM_FILENAME],
                    sample_batch["rewards"].max(),
                    len(sample_batch),
                    sample_batch["actions"],
                )
                if sample_batch["rewards"].max() == 1.0:
                    propagate_reward(sample_batch)
            return sample_batch
