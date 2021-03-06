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
a wrapper over ``gym-saturation`` environment with nice box features
"""
from typing import Callable, List

import gym
import numpy as np
import orjson


def _pad_features(features: np.ndarray, features_num: int) -> np.ndarray:
    if features_num >= features.shape[0]:
        padded_features = np.pad(
            features,
            ((0, features_num - features.shape[0]), (0, 0)),
        )
    else:
        padded_features = features[:features_num]
    return padded_features


class CustomFeatures(gym.ObservationWrapper):
    """a box wrapper for ``SaturationEnv``"""

    def __init__(
        self,
        env,
        clause_encoder: Callable[[dict], np.ndarray],
        features_num: int,
    ):
        super().__init__(env)
        observation_space = self.observation_space  # type: ignore
        avail_actions = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                observation_space["action_mask"].shape[0],
                features_num,
            ),
        )
        self.transforming_function = clause_encoder
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": observation_space["action_mask"],
                "avail_actions": avail_actions,
            }
        )
        self.encoded_state: List[np.ndarray] = []

    def observation(self, observation):
        new_clauses = list(
            map(
                orjson.loads,
                observation["real_obs"][len(self.encoded_state) :],
            )
        )
        new_embeddings = map(self.transforming_function, new_clauses)
        self.encoded_state += list(new_embeddings)
        padded_features = _pad_features(
            features=np.array(self.encoded_state),
            features_num=len(observation["action_mask"]),
        )
        return {
            "action_mask": observation["action_mask"],
            "avail_actions": padded_features,
        }
