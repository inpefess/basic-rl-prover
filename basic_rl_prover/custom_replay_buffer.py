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
# noqa: D205, D400
"""
Custom Replay Buffer
=====================
"""
import logging
import os
import sys
from typing import Optional, Set, Union

from gym_saturation.envs.saturation_env import PROBLEM_FILENAME
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


class CustomReplayBuffer(ReplayBuffer):
    """
    A custom replay buffer.

    >>> replay_buffer = CustomReplayBuffer()
    >>> sample_batch = getfixture("sample_batch") # noqa: F821
    >>> replay_buffer.add(sample_batch)
    >>> replay_buffer.stats()["num_entries"]
    2
    >>> sample = replay_buffer.sample(1000)
    >>> max(sample["rewards"])
    1.0
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
        self._problems_solved: Set[str] = set()

    @override(ReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Add only the episodes with positive reward."""
        if (
            isinstance(batch, SampleBatch)
            and self.storage_unit == StorageUnit.TIMESTEPS
        ):
            episodes = batch.split_by_episode()
            for episode in episodes:
                logger.info(
                    "EPISODE %d: %s %.1f %d %s",
                    episode[SampleBatch.EPS_ID][0],
                    os.path.basename(
                        episode[SampleBatch.INFOS][0][PROBLEM_FILENAME]
                    ),
                    episode[SampleBatch.REWARDS].sum(),
                    episode.count,
                    episode[SampleBatch.ACTIONS],
                )
                if (
                    episode[SampleBatch.REWARDS].max() > 0
                    and episode[SampleBatch.INFOS][0][PROBLEM_FILENAME]
                    not in self._problems_solved
                ):
                    self._problems_solved.add(
                        episode[SampleBatch.INFOS][0][PROBLEM_FILENAME]
                    )
                    timesteps = episode.timeslices(1)
                    for timestep in timesteps:
                        self._add_single_batch(timestep, **kwargs)

    def sample(self, num_items: int, **kwargs) -> Optional[SampleBatchType]:
        """
        Sample ``num_items`` items from this buffer.

        :param num_items: Number of items to sample from this buffer.
        :param kwargs: Forward compatibility kwargs.
        :returns: Concatenated batch of items.
        """
        if len(self) == 0:
            return SampleBatch({})
        return super().sample(num_items, **kwargs)
