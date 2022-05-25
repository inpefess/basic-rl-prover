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
Trained Prover Evaluation
=========================
"""
import os
from typing import List, Tuple

import gym
from ray.tune.analysis import ExperimentAnalysis
from torch.nn import Softmax

from basic_rl_prover.custom_dqn_trainer import CustomDQNTrainer
from basic_rl_prover.size_age_environment import size_age_env_creator
from basic_rl_prover.train_prover import get_config


def get_agent_and_env(
    problem_list: List[str],
) -> Tuple[CustomDQNTrainer, gym.Wrapper]:
    """
    read an agent from the best checkpoint

    :param problem_list: a list of filenames of TPTP problems
    :returns: a trained agent and and the environment in which it was trained
    """
    analysis = ExperimentAnalysis(
        os.path.join(os.environ["WORK"], "ray_results", "basic_rl_prover"),
        default_metric="episode_reward_mean",
        default_mode="max",
    )
    agent = CustomDQNTrainer(
        config=get_config(problem_list, {"num_workers": 1})
    )
    agent.restore(analysis.best_checkpoint)
    print(Softmax(1)(list(agent.get_policy().model.parameters())[0]))
    env = agent.env_creator(agent.get_config()["env_config"])
    return agent, env


def upload_and_test_agent(problem_list: List[str]) -> None:
    """
    upload and test agent

    >>> # to reproduce main results
    >>> from basic_rl_prover.constants import TRAIN_PROBLEMS
    >>> upload_and_test_agent(TRAIN_PROBLEMS)  # doctest: +SKIP

    :param problem_list: a list of filenames of TPTP problems
    :returns:
    """
    agent, env = get_agent_and_env(problem_list)
    for filename in problem_list:
        env = gym.wrappers.TimeLimit(
            size_age_env_creator(
                {"problem_list": [filename], "max_clauses": 1000}
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
