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


def evaluate_baseline(problem_list: List[str]) -> None:
    """
    baseline evaluation

    >>> from importlib.resources import files
    >>> problem_filename = os.path.join(
    ...     files("basic_rl_prover")
    ...     .joinpath(os.path.join(
    ...         "resources", "TPTP-mock", "Problems", "TST", "TST001-1.p"
    ...     ))
    ... )
    >>> evaluate_baseline([problem_filename])
    TST001-1.p 1.0 3

    :param problem_list: a list of filenames of TPTP problems
    :returns:
    """
    for filename in problem_list:
        env = gym.wrappers.TimeLimit(
            gym.make(
                "GymSaturation-v0", problem_list=[filename], max_clauses=1000
            ),
            max_episode_steps=100,
        )
        print(
            os.path.basename(filename),
            episode(env, SizeAgeAgent(5, 1)).reward,
            # pylint: disable=protected-access
            env._elapsed_steps,
            flush=True,
        )
