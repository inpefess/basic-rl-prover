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
a wrapper over ``gym-saturation`` environment using ast2vec model to embed
logical clauses
"""
import json
from operator import itemgetter
from typing import List
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import gymnasium as gym
import numpy as np


def _pad_features(features: np.ndarray, features_num: int) -> np.ndarray:
    if features_num >= features.shape[0]:
        padded_features = np.pad(
            features,
            ((0, features_num - features.shape[0]), (0, 0)),
        )
    else:
        padded_features = features[:features_num]
    return padded_features


class AST2VecFeatures(gym.Wrapper):
    """A box wrapper for ``SaturationEnv``."""

    _torch_serve_url = "http://127.0.0.1:9080/predictions/ast2vec"

    def __init__(
        self,
        env,
        features_num: int,
    ):
        """Initialise all the things."""
        super().__init__(env)
        action_mask = gym.spaces.Box(
            low=0, high=1, shape=(env.action_space.n, 1)
        )
        avail_actions = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                action_mask.shape[0],
                features_num,
            ),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": action_mask,
                "avail_actions": avail_actions,
            }
        )
        self.encoded_state: List[np.ndarray] = []

    def reset(self, **kwargs):
        """Reset the environment."""
        observation, info = self.env.reset(**kwargs)
        self.encoded_state = []
        info["real_obs"] = observation
        return self._transform(observation), info

    def _transform(self, observation):
        new_clauses = [
            clause["literals"]
            for clause in observation[len(self.encoded_state) :]
        ]
        new_embeddings = map(self.ast2vec_features, new_clauses)
        self.encoded_state += list(new_embeddings)
        padded_features = _pad_features(
            features=np.array(self.encoded_state),
            features_num=self.observation_space["action_mask"].shape[0],
        )
        return {
            "action_mask": np.expand_dims(
                np.pad(
                    1
                    - np.array(
                        list(map(itemgetter("processed"), observation))
                    ),
                    (
                        0,
                        self.observation_space["action_mask"].shape[0]
                        - len(observation),
                    ),
                ),
                axis=1,
            ),
            "avail_actions": padded_features,
        }

    def step(self, action):
        """Apply the agent's action."""
        observation, reward, terminated, truncated, info = self.env.step(
            action
        )
        info["real_obs"] = observation
        try:
            return (
                self._transform(observation),
                reward,
                terminated,
                truncated,
                info,
            )
        except HTTPError as ast2vec_error:
            if ast2vec_error.code == 507:  # Insufficient Storage
                return (
                    {
                        "action_mask": np.zeros_like(
                            self.observation_space["action_mask"]
                        ),
                        "avail_actions": np.zeros_like(
                            self.observation_space["avail_actions"]
                        ),
                    },
                    -1.0,
                    False,
                    True,
                    info,
                )
            raise ast2vec_error

    def ast2vec_features(self, literals_str: str) -> dict:
        """
        Encode literals using ast2vec.

        :param literals_str: literals to encode
        :returns: observation dict with ast2vec encoding instead of clauses
        """
        prepared_literals = (
            literals_str.replace("==", "^^")
            .replace("!=", "^^^")
            .replace("=", "==")
            .replace("^^^", "!=")
            .replace("^^", "==")
            .replace("$false", "False")
            .replace("as", "__as")
        )
        req = Request(
            self._torch_serve_url,
            f'{{"data": "{prepared_literals}"}}'.encode("utf8"),
            {"Content-Type": "application/json"},
        )
        with urlopen(req) as response:
            clause_embedding = json.loads(response.read().decode("utf-8"))
        return clause_embedding


def ast2vec_env_creator(env_config: dict) -> gym.Wrapper:
    """
    Create the wrapped environment.

    >>> import os
    >>> from glob import glob
    >>> from importlib.resources import files
    >>> problem_list = sorted(glob(os.path.join(
    ...     files("gym_saturation").joinpath("resources"),
    ...     "TPTP-mock", "Problems", "*", "*-*.p"
    ...     )
    ... ))
    >>> env = ast2vec_env_creator({"problem_list": problem_list})
    >>> env.observation_space["avail_actions"].shape[1]
    256

    :param env_config: a custom environment config
    :returns: a ``SaturationEnv``  with ast2vec encoding
    """
    env = gym.make("Vampire-v0", **env_config)
    return AST2VecFeatures(env, 256)
