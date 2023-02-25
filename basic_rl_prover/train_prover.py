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
Training an RL Prover
=====================
"""
import os
from typing import Any, Dict, List, Optional

import ray
from ray.air.config import CheckpointConfig, RunConfig
from ray.rllib.algorithms.apex_dqn import ApexDQNConfig
from ray.tune import TuneConfig, Tuner
from ray.tune.logger import CSVLoggerCallback
from ray.tune.registry import register_env

from basic_rl_prover.action_selection_model import ActionSelectionModel
from basic_rl_prover.ast2vec_environment import ast2vec_env_creator
from basic_rl_prover.custom_callbacks import CustomCallbacks
from basic_rl_prover.custom_dqn import CustomDQN
from basic_rl_prover.custom_replay_buffer import CustomReplayBuffer


def _set_other_parameters(config: ApexDQNConfig) -> None:
    config.framework("torch")
    config.resources(num_gpus=0)
    config.exploration(explore=True)
    config.debugging(seed=17)
    config.reporting(min_sample_timesteps_per_iteration=1)
    config.callbacks(CustomCallbacks)


def get_config(problem_list: List[str]) -> ApexDQNConfig:
    """
    Get a prepacked config.

    :param problem_list: a list of filenames of TPTP problems
    :returns: a config
    """
    register_env("ast2vec_saturation", ast2vec_env_creator)
    env_config = {"problem_list": problem_list, "max_clauses": 1000}
    config = ApexDQNConfig()
    config.training(
        model={
            "custom_model": ActionSelectionModel,
            "custom_model_config": {"embedding_size": 256},
        },
        hiddens=[],  # type:ignore
        dueling=False,
        lr=0.01,
        num_steps_sampled_before_learning_starts=1,
        train_batch_size=1024,
    )
    config.replay_buffer_config.update(
        {
            "type": CustomReplayBuffer,
            "capacity": 10000,
            "worker_side_prioritization": False,
        }
    )
    config.environment(
        env="ast2vec_saturation",
        env_config=env_config,
        disable_env_checking=True,
    )
    config.rollouts(
        batch_mode="complete_episodes",
        num_rollout_workers=2,
    )
    _set_other_parameters(config)
    return config


def train_a_prover(
    problem_list: List[str],
    stop: Optional[Dict[str, int]] = None,
    custom_config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Run Ray pipeline.

    >>> os.environ["WORK"] = getfixture("tmp_path").as_posix() # noqa: F821
    >>> from importlib.resources import files
    >>> problem_filename = os.path.join(
    ...     files("gym_saturation")
    ...     .joinpath(os.path.join(
    ...         "resources", "TPTP-mock", "Problems", "TST", "TST003-1.p"
    ...     ))
    ... )
    >>> # the shortest path to the proof here is [0, 1]
    >>> # 2 is added after 0 because of exploration
    >>> # and all three actions contribute to the proof
    >>> train_a_prover(
    ...     [problem_filename],
    ...     {"training_iteration": 1},
    ...     {
    ...         "train_batch_size": 1,
    ...         "num_rollout_workers": 1,
    ...     }
    ... )
    == Status ==
    ... TST003-1.p 3 3 [1 2 0]
    ...
    >>> from basic_rl_prover.test_prover import upload_and_test_agent
    >>> # the actual proof consists of five steps though: 0, 1, 2, 0+1,
    >>> # and final $false. They are selected automatically by Vampire
    >>> upload_and_test_agent([problem_filename])
    TST003-1.p 1.0 5 2 [0, 1]

    :param problem_list: a list of filenames of TPTP problems
    :param stop: `a stop condition <https://docs.ray.io/en/latest/tune/tutorials/tune-stopping.html#stopping-with-a-dictionary>`_
    :param custom_config: additional parameters to change in the default config
    :returns:
    """
    ray.init(ignore_reinit_error=True)
    full_config = dict(get_config(problem_list).to_dict())
    if custom_config is not None:
        full_config.update(custom_config)
    run_config = RunConfig(
        name="basic_rl_prover",
        local_dir=os.path.join(os.environ["WORK"], "ray_results"),
        checkpoint_config=CheckpointConfig(checkpoint_frequency=1),
        stop=stop,
        callbacks=[CSVLoggerCallback()],
    )
    tune_config = TuneConfig(time_budget_s=3600)
    Tuner(
        trainable=CustomDQN,
        param_space=full_config,
        run_config=run_config,
        tune_config=tune_config,
    ).fit()
    ray.shutdown()
