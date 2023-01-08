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
Custom Replay Buffer
=====================
"""
import logging
import os
import sys
from typing import Optional, Union

import numpy as np
from gym_saturation.envs.saturation_env import PROBLEM_FILENAME
from ray.rllib.policy.sample_batch import (
    MultiAgentBatch,
    SampleBatch,
    concat_samples,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.replay_buffer import (
    ReplayBuffer,
    StorageUnit,
)
from ray.rllib.utils.typing import SampleBatchType

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def filter_batch(
    batch_to_filter: SampleBatch, indices: np.ndarray
) -> SampleBatch:
    """
    Filter only specified indices from a batch.

    :param batch_to_filter: a batch to filter
    :param indices: indices to leave (Boolean array)
    :returns: a new batch with only specified indices
    """
    return SampleBatch(
        {key: batch_to_filter[key][indices] for key in batch_to_filter.keys()}
    )


class CustomReplayBuffer(ReplayBuffer):
    """
    A custom replay buffer.

    >>> from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
    >>> replay_buffer = CustomReplayBuffer()
    >>> sample_batch = getfixture("sample_batch") # noqa: F821
    >>> replay_buffer.sample(1000).env_steps()
    0
    >>> replay_buffer.add(sample_batch)
    >>> sample = replay_buffer.sample(1000)[DEFAULT_POLICY_ID]
    >>> sample["rewards"].sum()
    500.0
    >>> set(sample["actions"])
    {0, 1}
    >>> set(sample[SampleBatch.EPS_ID])
    {1}
    """

    def __init__(
        self,
        capacity: int = 10000,
        storage_unit: Union[str, StorageUnit] = "timesteps",
        **kwargs,
    ):
        """Initialise a CustomReplayBuffer instance.

        :param capacity: Max number of timesteps to store in this FIFO
            buffer. After reaching this number, older samples will be
            dropped to make space for new ones.
        :param storage_unit: If not a StorageUnit, either 'timesteps',
            'sequences' or 'episodes'. Specifies how experiences are stored.
        :param kwargs: Forward compatibility kwargs.
        """
        super().__init__(capacity, storage_unit, **kwargs)
        positive_capacity = capacity // 2
        self.positive_buffer = ReplayBuffer(
            positive_capacity, StorageUnit.TIMESTEPS
        )
        self.non_positive_buffer = ReplayBuffer(
            capacity - positive_capacity, StorageUnit.TIMESTEPS
        )

    @override(ReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Add to a sub-buffer depending on the reward."""
        if isinstance(batch, SampleBatch):
            episodes = batch.split_by_episode()
            for episode in episodes:
                positive_action_indices = episode[SampleBatch.REWARDS] > 0
                num_positive_actions = sum(positive_action_indices)
                logger.info(
                    "EPISODE %d: %s %d %d %s",
                    episode[SampleBatch.EPS_ID][0],
                    os.path.basename(
                        episode[SampleBatch.INFOS][0][PROBLEM_FILENAME]
                    ),
                    num_positive_actions,
                    episode.count,
                    episode[SampleBatch.ACTIONS],
                )
                self.positive_buffer.add(
                    filter_batch(episode, positive_action_indices)
                )
                self.non_positive_buffer.add(
                    filter_batch(episode, ~positive_action_indices)
                )

    def sample(self, num_items: int, **kwargs) -> Optional[SampleBatchType]:
        """
        Sample ``num_items`` items from this buffer.

        :param num_items: Number of items to sample from this buffer.
        :param kwargs: Forward compatibility kwargs.
        :returns: Concatenated batch of items.
        """
        has_positives = len(self.positive_buffer) > 0
        has_non_positives = len(self.non_positive_buffer) > 0
        if not has_positives and not has_non_positives:
            return MultiAgentBatch({}, 0)
        if has_positives and has_non_positives:
            num_positive_items = num_items // 2
            return concat_samples(
                [
                    self.positive_buffer.sample(num_positive_items),
                    self.non_positive_buffer.sample(
                        num_items - num_positive_items
                    ),
                ]
            ).as_multi_agent()
        if has_positives:
            return self.positive_buffer.sample(num_items).as_multi_agent()
        return self.non_positive_buffer.sample(num_items).as_multi_agent()

    def update_priorities(self, prio_dict: dict) -> None:
        """Ape-X supposes the buffer is prioritised."""
