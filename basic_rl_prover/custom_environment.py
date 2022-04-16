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
import orjson
from gym.wrappers import TransformObservation
from gym_saturation.logic_ops.utils import clause_length


def age_size_role_features(observation: dict) -> dict:
    """
    >>> import numpy as np
    >>> from gym_saturation.clause_space import ClauseSpace
    >>> observation = {
    ...     "real_obs": map(orjson.dumps, 2 * ClauseSpace().sample()),
    ...     "action_mask": np.array([1, 0])
    ... }
    >>> age_size_role_features(observation)  # doctest: +ELLIPSIS
    {'action_mask': array([1, 0]), 'avail_actions': array([[1. ..., 1. ...],
           [0.36787945, 0.36787945]], dtype=float32)}
    >>> observation = {
    ...     "real_obs": map(orjson.dumps, 20 * ClauseSpace().sample()),
    ...     "action_mask": np.array([1, 0])
    ... }
    >>> age_size_role_features(observation)  # doctest: +ELLIPSIS
    {'a...rray([1, 0]), 'avail_actions': array([[1.0000000e+00, 1.0000000e+00],
           [4.1399378e-08, 4.1399378e-08]], dtype=float32)}

    :param observation: an observation dict from ``SaturationEnv``
    :returns: observation dict with age and size features instead of clauses
    """
    clauses = list(map(orjson.loads, observation["real_obs"]))
    features = np.array(
        [[clause_length(clause), clause["birth_step"]] for clause in clauses],
        np.float32,
    )
    features = np.exp(-features.argsort(axis=0).argsort(axis=0)).astype(
        np.float32
    )
    features_num = len(observation["action_mask"])
    if features_num >= features.shape[0]:
        padded_features = np.pad(
            features,
            ((0, features_num - features.shape[0]), (0, 0)),
        )
    else:
        padded_features = features[:features_num]
    return {
        "action_mask": observation["action_mask"],
        "avail_actions": padded_features,
    }


class AgeSizeFeatures(TransformObservation):
    """a wrapper adding age and size features to ``SaturationEnv``"""

    def __init__(self, env, f):
        super().__init__(env, f)
        avail_actions = gym.spaces.Box(
            low=0,
            high=np.infty,
            shape=(self.observation_space["action_mask"].shape[0], 2),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": self.observation_space["action_mask"],
                "avail_actions": avail_actions,
            }
        )


def custom_env_creator(env_config: dict) -> gym.Wrapper:
    """
    >>> env = custom_env_creator({"problem_list": []})
    >>> env.observation_space["avail_actions"].shape[1]
    2
    >>> env = custom_env_creator(
    ...     {"problem_list": [], "vampire_binary_path": "vampire"}
    ... )
    >>> env.observation_space["avail_actions"].shape[1]
    2

    :param env_config: a custom environment config
    :returns: a ``SaturationEnv`` or ``VampireEnv`` (if ``vampire_binary_path``
        key is present in ``env_config``) with age and size features
    """
    if "vampire_binary_path" in env_config:
        env = gym.make("GymVampire-v0", **env_config)
    else:
        env = gym.make("GymSaturation-v0", **env_config)
    return AgeSizeFeatures(env, age_size_role_features)
