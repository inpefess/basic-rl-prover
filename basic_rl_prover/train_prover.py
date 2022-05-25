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
Traning an RL Prover
====================
"""
import os
from typing import Any, Dict, List, Optional

import ray
from ray import tune
from ray.tune.registry import register_env

from basic_rl_prover.action_selection_model import ActionSelectionModel
from basic_rl_prover.custom_dqn_trainer import CustomDQNTrainer
from basic_rl_prover.size_age_environment import size_age_env_creator


def get_config(
    problem_list: List[str],
    custom_env_config: Optional[Dict[str, int]],
    vampire_binary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    :param problem_list: a list of filenames of TPTP problems
    :param custom_config: additional parameters to change in the default config
    :param vampire_binary_path: a full path to Vampire binary
    :returns: a config
    """
    register_env("age_size_saturation", size_age_env_creator)
    env_config = {"problem_list": problem_list, "max_clauses": 1000}
    if vampire_binary_path is not None:
        env_config["vampire_binary_path"] = vampire_binary_path
    basic_config = {
        "seed": 777,
        "env": "age_size_saturation",
        "env_config": env_config,
        "framework": "torch",
        "model": {"custom_model": ActionSelectionModel},
        "batch_mode": "complete_episodes",
        "horizon": 100,
        "num_workers": 10,
        "hiddens": [],
        "dueling": False,
        "learning_starts": 1,
        "lr": 0.01,
        "disable_env_checking": True,
    }
    if custom_env_config is not None:
        basic_config.update(custom_env_config)
    return basic_config


def train_a_prover(
    problem_list: List[str],
    stop: Optional[Dict[str, int]] = None,
    custom_config: Optional[Dict[str, int]] = None,
    vampire_binary_path: Optional[str] = None,
) -> None:
    """
    run ray pipeline

    >>> os.environ["WORK"] = "."
    >>> import shutil
    >>> shutil.rmtree("ray_results", ignore_errors=True)
    >>> from importlib.resources import files
    >>> problem_filename = os.path.join(
    ...     files("gym_saturation")
    ...     .joinpath(os.path.join(
    ...         "resources", "TPTP-mock", "Problems", "TST", "TST003-1.p"
    ...     ))
    ... )
    >>> # this takes several seconds
    >>> train_a_prover(
    ...     [problem_filename],
    ...     {"training_iteration": 1},
    ...     {
    ...         "timesteps_per_iteration": 1,
    ...         "train_batch_size": 1,
    ...         "num_workers": 1,
    ...     },
    ... )  # doctest: +ELLIPSIS
    == Status ==
    .../resources/TPTP-mock/Problems/TST/TST003-1.p 1 4 [0 1 2 3]
    ...
    >>> from basic_rl_prover.test_prover import upload_and_test_agent
    >>> upload_and_test_agent([problem_filename])
    tensor([[0.2650, 0.7350]], grad_fn=<SoftmaxBackward0>)
    TST003-1.p 1.0 4 [0, 1, 2, 3]
    >>> # to reproduce the results
    >>> from basic_rl_prover.constants import TRAIN_PROBLEMS
    >>> train_a_prover(TRAIN_PROBLEMS, None, None)  # doctest: +SKIP
    >>> # test with Vampire backend
    >>> shutil.rmtree("ray_results", ignore_errors=True)
    >>> train_a_prover(
    ...     [problem_filename],
    ...     {"training_iteration": 1},
    ...     {
    ...         "timesteps_per_iteration": 1,
    ...         "train_batch_size": 1,
    ...         "num_workers": 1,
    ...     },
    ...     "vampire",
    ... )  # doctest: +ELLIPSIS
    == Status ==
    .../resources/TPTP-mock/Problems/TST/TST003-1.p 1 3 [2 1 0]
    ...

    :param problem_list: a list of filenames of TPTP problems
    :param stop: `a stop condition <https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html#stopping-with-a-dictionary>`_
    :param custom_config: additional parameters to change in the default config
    :param vampire_binary_path: a full path to Vampire binary
    :returns:
    """
    ray.init(ignore_reinit_error=True)
    tune.run(
        CustomDQNTrainer,
        name="basic_rl_prover",
        config=get_config(problem_list, custom_config, vampire_binary_path),
        local_dir=os.path.join(os.environ["WORK"], "ray_results"),
        checkpoint_freq=1,
        time_budget_s=3600,
        stop=stop,
    )
    ray.shutdown()
