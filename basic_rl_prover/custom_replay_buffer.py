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
Custom Replay Buffer
=====================
"""
import logging
import sys

from gym_saturation.envs.saturation_env import (
    POSITIVE_ACTIONS,
    PROBLEM_FILENAME,
    STATE_DIFF_UPDATED,
)
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.replay_buffer import (
    ReplayBuffer,
    StorageUnit,
)
from ray.rllib.utils.typing import SampleBatchType

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
        for info in sample_batch[SampleBatch.INFOS]
        if POSITIVE_ACTIONS in info
    ][0]
    proof_length = len(positive_actions)
    new_rewards = sample_batch[SampleBatch.REWARDS].copy()
    for i, action in enumerate(sample_batch[SampleBatch.ACTIONS]):
        if action in positive_actions:
            new_rewards[i] = 1.0 / proof_length
    sample_batch[SampleBatch.REWARDS] = new_rewards
    for info in sample_batch[SampleBatch.INFOS]:
        info.pop(STATE_DIFF_UPDATED)
        if POSITIVE_ACTIONS in info:
            info.pop(POSITIVE_ACTIONS)


class CustomReplayBuffer(ReplayBuffer):
    """
    >>> import numpy as np
    >>> from gym.spaces import Box, Discrete
    >>> clause1 = {"class": "Clause", "processed": True, "literals": [], "label": "this_is_a_test_case", "birth_step": 1, "inference_parents": ["initial"], "inference_rule": "success"}
    >>> clause0 = {"class": "Clause", "processed": True, "literals": [{"class": "Literal", "negated": False, "atom": {"class": "Predicate", "name": "this_is_a_test_case", "arguments": []}, }], "label": "initial", "birth_step": 0, "inference_parents": None, "inference_rule": None}
    >>> replay_buffer = CustomReplayBuffer()
    >>> two_trajectories = SampleBatch(
    ...     infos=[
    ...         {
    ...             STATE_DIFF_UPDATED: {0: clause0},
    ...             PROBLEM_FILENAME: "test"
    ...         },
    ...         {
    ...             STATE_DIFF_UPDATED: {1: clause1},
    ...             PROBLEM_FILENAME: "test",
    ...         },
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
    ...     rewards=np.array([0.0, 0.0, 0.0, 1.0]),
    ...     actions=np.array([0, 1, 0, 1]),
    ...     dones=np.array([False, True, False, True]),
    ...     eps_id=np.array([0, 0, 1, 1]),
    ... )
    >>> replay_buffer.add(two_trajectories)
    >>> replay_buffer.stats()["num_entries"]
    2
    >>> sample = replay_buffer.sample(1000)
    >>> set(sample["rewards"])
    {0.5}
    >>> set(sample["actions"])
    {0, 1}
    >>> set(sample[SampleBatch.EPS_ID])
    {1}
    """

    @override(ReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        if (
            isinstance(batch, SampleBatch)
            and self.storage_unit == StorageUnit.TIMESTEPS
        ):
            episodes = batch.split_by_episode()
            for episode in episodes:
                logger.info(
                    "EPISODE %d: %s %d %d %s",
                    episode[SampleBatch.EPS_ID][0],
                    episode[SampleBatch.INFOS][0][PROBLEM_FILENAME],
                    episode[SampleBatch.REWARDS].max(),
                    episode.count,
                    episode[SampleBatch.ACTIONS],
                )
                if episode[SampleBatch.REWARDS].max() > 0:
                    propagate_reward(episode)
                    timesteps = episode.timeslices(1)
                    for timestep in timesteps:
                        self._add_single_batch(timestep, **kwargs)