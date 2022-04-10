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
from typing import List, Optional

import gym
from gym_saturation.agent_testing import SizeAgeAgent, episode
from gym_saturation.envs import SaturationEnv
from gym_saturation.logic_ops.utils import WrongRefutationProofError


def _proof_found(env: SaturationEnv) -> bool:
    try:
        return env.tstp_proof is not None
    except WrongRefutationProofError:
        return False


def evaluate_baseline(
    problem_list: List[str],
    max_episode_steps: int,
    vampire_binary_path: Optional[str] = None,
) -> None:
    """
    baseline evaluation

    >>> from importlib.resources import files
    >>> problem_filename = os.path.join(
    ...     files("gym_saturation")
    ...     .joinpath(os.path.join(
    ...         "resources", "TPTP-mock", "Problems", "TST", "TST003-1.p"
    ...     ))
    ... )
    >>> evaluate_baseline([problem_filename], 100)
    TST003-1.p True 4
    >>> evaluate_baseline([problem_filename], 1)
    TST003-1.p False 1
    >>> evaluate_baseline([problem_filename], 100, "vampire")
    TST003-1.p True 3

    :param problem_list: a list of filenames of TPTP problems
    :param max_episode_steps: a maximal number of saturation algorithm steps
    :param vampire_binary_path: a full path to Vampire prover binary.
        If not specified, a default Python implementation is used instead
    :returns:
    """
    for filename in problem_list:
        if vampire_binary_path is None:
            basic_env = gym.make(
                "GymSaturation-v0", problem_list=[filename], max_clauses=1000
            )
        else:
            basic_env = gym.make(
                "GymVampire-v0",
                problem_list=[filename],
                max_clauses=1000,
                vampire_binary_path=vampire_binary_path,
            )
        env = gym.wrappers.TimeLimit(
            basic_env,
            max_episode_steps=max_episode_steps,
        )
        episode(env, SizeAgeAgent(5, 1))
        success = _proof_found(env)
        print(
            os.path.basename(filename),
            success,
            # pylint: disable=protected-access
            env._elapsed_steps,
            flush=True,
        )
