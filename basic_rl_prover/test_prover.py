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
Trained Prover Evaluation
=========================
"""
import os
from typing import List

import gym
from ray.tune.analysis import ExperimentAnalysis

from basic_rl_prover.ast2vec_environment import ast2vec_env_creator
from basic_rl_prover.custom_dqn import CustomDQN
from basic_rl_prover.train_prover import get_config


def get_agent(problem_list: List[str], vampire_binary_path: str) -> CustomDQN:
    """
    Read an agent from the best checkpoint.

    :param problem_list: a list of filenames of TPTP problems
    :param vampire_binary_path: an absolute path to Vampire binary
    :returns: a trained agent
    """
    analysis = ExperimentAnalysis(
        os.path.join(os.environ["WORK"], "ray_results", "basic_rl_prover"),
        default_metric="episodes_total",
        default_mode="max",
    )
    dqn_config = get_config(problem_list, vampire_binary_path)
    dqn_config.rollouts(num_rollout_workers=0)
    dqn_config.resources(num_gpus=0)
    agent = CustomDQN(config=dqn_config)
    agent.restore(analysis.best_checkpoint)
    return agent


def upload_and_test_agent(problem_list: List[str]) -> None:
    """
    Upload and test agent.

    >>> # to reproduce main results
    >>> from basic_rl_prover.constants import TRAIN_PROBLEMS
    >>> upload_and_test_agent(TRAIN_PROBLEMS)  # doctest: +SKIP

    :param problem_list: a list of filenames of TPTP problems
    :returns:
    """
    agent = get_agent(problem_list, "vampire")
    for filename in problem_list:
        env = gym.wrappers.TimeLimit(
            ast2vec_env_creator(
                {
                    "problem_list": [filename],
                    "max_clauses": 500,
                    "vampire_binary_path": "vampire",
                }
            ),
            max_episode_steps=100,
        )
        obs, done, actions = env.reset(), False, []
        while not done:
            action = agent.compute_single_action(obs, explore=False)
            actions.append(action)
            obs, reward, done, _ = env.step(action)
        print(
            os.path.basename(filename),
            reward,
            len(actions),
            actions,
            flush=True,
        )
