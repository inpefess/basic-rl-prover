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
import re
import shutil
import sys
from hashlib import sha256
from typing import Dict, List, Optional, Union

import numpy as np
from gym_saturation.envs.saturation_env import PROBLEM_FILENAME
from gym_saturation.utils import Clause
from ray.rllib.policy.sample_batch import SampleBatch, concat_samples
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.replay_buffer import (
    ReplayBuffer,
    StorageUnit,
)
from ray.rllib.utils.typing import SampleBatchType

GENERATED_PROBLEMS_DIR = os.path.join(
    "/", "home", "boris", "generated_problems"
)

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


def is_trivial_tautology(clause: Clause) -> bool:
    """
    Check whether a clause contains a literal and its negation.

    :param clause: a TPTP clause
    :returns: whether a clause is a trivial tautology or not
    """
    literals = clause.literals.split("|")
    literals_count = len(literals)
    for i, literal in enumerate(literals):
        for j in range(i + 1, literals_count):
            literal_one = literal.replace(" ", "")
            literal_two = literals[j].replace(" ", "")
            if (
                literal_one[0] == "~"
                and literal_one[1:] == literal_two
                or literal_two[0] == "~"
                and literal_two[1:] == literal_one
            ):
                return True
    return False


def negate_clause(clause: Clause) -> List[Clause]:
    """
    Transform a negation of a clause into list of clauses in CNF.

    :param clause: a clause in TPTP syntax
    :returns: a list of clauses in TPTP syntax.
    """
    skolemised_literals = re.sub(
        r"X(\d+)", r"generated_symbol_\1", clause.literals
    )
    literals = skolemised_literals.split("|")
    new_clauses = []
    for literal in literals:
        if "~" in literal:
            new_clauses.append(Clause(literals=literal.replace("~", "")))
        else:
            new_clauses.append(Clause(literals=f"~{literal}"))
    return new_clauses


def generate_problems(final_state: Dict[str, Clause]) -> None:
    """
    Generate TPTP problems related to failed one.

    :param final_state: the final state of a failed proof attempts
    """
    original_clauses = "\n".join(
        [
            f"cnf({label},plain,{clause.literals})."
            for label, clause in final_state.items()
            if clause.inference_rule == "input"
        ]
    )
    generated_non_trivial_clauses = [
        clause
        for clause in final_state.values()
        if clause.inference_rule != "input"
        and not is_trivial_tautology(clause)
    ]
    shutil.rmtree(GENERATED_PROBLEMS_DIR, ignore_errors=True)
    os.mkdir(GENERATED_PROBLEMS_DIR)
    for generated_clause in generated_non_trivial_clauses:
        problem_text = "\n".join(
            [original_clauses]
            + [
                f"cnf({clause.label},plain,{clause.literals})."
                for clause in negate_clause(generated_clause)
            ]
        )
        problem_filename = os.path.join(
            GENERATED_PROBLEMS_DIR,
            f"generated{sha256(problem_text.encode('utf8')).hexdigest()}.p",
        )
        with open(
            problem_filename,
            "w",
            encoding="utf8",
        ) as problem_file:
            problem_file.write(problem_text)


class CustomReplayBuffer(ReplayBuffer):
    """
    A custom replay buffer.

    >>> replay_buffer = CustomReplayBuffer()
    >>> sample_batch = getfixture("sample_batch") # noqa: F821
    >>> replay_buffer.add(sample_batch)
    >>> sample = replay_buffer.sample(1000)
    >>> sample["rewards"].sum()
    500.0
    >>> set(sample["actions"])
    {0, 1}
    >>> set(sample[SampleBatch.EPS_ID])
    {0, 1}
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
        self.negative_buffer = ReplayBuffer(
            capacity - positive_capacity, StorageUnit.TIMESTEPS
        )

    @override(ReplayBuffer)
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Add to a sub-buffer depending on the reward."""
        if isinstance(batch, SampleBatch):
            episodes = batch.split_by_episode()
            for episode in episodes:
                positive_action_indices = episode[SampleBatch.REWARDS] > 0
                logger.info(
                    "EPISODE %d: %s %d %d %s",
                    episode[SampleBatch.EPS_ID][0],
                    os.path.basename(
                        episode[SampleBatch.INFOS][0][PROBLEM_FILENAME]
                    ),
                    positive_action_indices.sum(),
                    episode.count,
                    episode[SampleBatch.ACTIONS],
                )
                if episode[SampleBatch.REWARDS].sum() > 0:
                    self.positive_buffer.add(
                        filter_batch(episode, positive_action_indices)
                    )
                    self.negative_buffer.add(
                        filter_batch(episode, ~positive_action_indices)
                    )
                elif (
                    not "generated"
                    in episode[SampleBatch.INFOS][-1]["problem_filename"]
                ):
                    generate_problems(
                        episode[SampleBatch.INFOS][-1]["real_obs"]
                    )

    def sample(self, num_items: int, **kwargs) -> Optional[SampleBatchType]:
        """
        Sample ``num_items`` items from this buffer.

        :param num_items: Number of items to sample from this buffer.
        :param kwargs: Forward compatibility kwargs.
        :returns: Concatenated batch of items.
        """
        if len(self.positive_buffer) == 0:
            return SampleBatch({})
        num_positive_items = num_items // 2
        return concat_samples(
            [
                self.positive_buffer.sample(num_positive_items),
                self.negative_buffer.sample(num_items - num_positive_items),
            ]
        )
