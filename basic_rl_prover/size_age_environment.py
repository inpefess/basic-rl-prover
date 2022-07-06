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
""" a customisation of a ``gym-saturation`` environment """
import gym
import numpy as np
from gym_saturation.logic_ops.utils import clause_length

from basic_rl_prover.custom_features import CustomFeatures


def size_age_features(clause: dict) -> np.ndarray:
    """
    >>> from gym_saturation.clause_space import ClauseSpace
    >>> import orjson
    >>> clause = orjson.loads(ClauseSpace().sample()[0])
    >>> size_age_features(clause)
    array([1., 1.], dtype=float32)

    :param clause: a clause in a JSON-like format
    :returns: age and size of the clause
    """
    return 1 / (
        1 + np.array([clause_length(clause), clause["birth_step"]], np.float32)
    )


def size_age_env_creator(env_config: dict) -> gym.Wrapper:
    """
    >>> import os
    >>> from glob import glob
    >>> from importlib.resources import files
    >>> problem_list = sorted(glob(os.path.join(
    ...     files("gym_saturation").joinpath("resources"),
    ...     "TPTP-mock", "Problems", "*", "*-*.p"
    ...     )
    ... ))
    >>> env = size_age_env_creator({"problem_list": problem_list})
    >>> env.observation_space["avail_actions"].shape[1]
    2
    >>> env = size_age_env_creator(
    ...     {"problem_list": problem_list, "vampire_binary_path": "vampire"}
    ... )
    >>> env.observation_space["avail_actions"].shape[1]
    2

    :param env_config: a custom environment config
    :returns: a ``SaturationEnv`` or ``VampireEnv`` (if ``vampire_binary_path``
        key is present in ``env_config``) with size and age features
    """
    if "vampire_binary_path" in env_config:
        env = gym.make("GymVampire-v0", **env_config)
    else:
        env = gym.make("GymSaturation-v0", **env_config)
    return CustomFeatures(env, size_age_features, 2)
