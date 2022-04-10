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
Heuristics Mixture Baseline
===========================
"""
import os
from typing import List

import gym
from gym_saturation.agent_testing import SizeAgeAgent, episode
from gym_saturation.logic_ops.utils import WrongRefutationProofError


def evaluate_baseline(problem_list: List[str], max_episode_steps: int) -> None:
    """
    baseline evaluation

    >>> from importlib.resources import files
    >>> problem_filename = os.path.join(
    ...     files("basic_rl_prover")
    ...     .joinpath(os.path.join(
    ...         "resources", "TPTP-mock", "Problems", "TST", "TST001-1.p"
    ...     ))
    ... )
    >>> evaluate_baseline([problem_filename], 100)
    TST001-1.p True 2
    >>> evaluate_baseline([problem_filename], 1)
    TST001-1.p False 1

    :param problem_list: a list of filenames of TPTP problems
    :param max_episode_steps: a maximal number of saturation algorithm steps
    :returns:
    """
    for filename in problem_list:
        env = gym.wrappers.TimeLimit(
            gym.make(
                "GymSaturation-v0", problem_list=[filename], max_clauses=1000
            ),
            max_episode_steps=max_episode_steps,
        )
        episode(env, SizeAgeAgent(5, 1))
        try:
            success = env.tstp_proof is not None
        except WrongRefutationProofError:
            success = False
        print(
            os.path.basename(filename),
            success,
            # pylint: disable=protected-access
            env._elapsed_steps,
            flush=True,
        )
