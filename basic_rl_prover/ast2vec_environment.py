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
a wrapper over ``gym-saturation`` environment using ast2vec model to embed
logical clauses
"""
from itertools import chain
from operator import itemgetter
from typing import List, Tuple
from urllib.request import Request, urlopen

import gym
import numpy as np
import orjson
from tptp_lark_parser.grammar import Clause, Function, Literal, Term
from tptp_lark_parser.tptp_parser import TPTPParser


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

    _torch_serve_url = "http://127.0.0.1:8080/predictions/ast2vec"

    def __init__(
        self,
        env,
        features_num: int,
    ):
        """Initialise all the things."""
        super().__init__(env)
        observation_space: gym.spaces.Dict = (
            self.observation_space  # type: ignore
        )
        avail_actions = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                observation_space["action_mask"].shape[0],
                features_num,
            ),
        )
        self.observation_space = gym.spaces.Dict(
            {
                "action_mask": observation_space["action_mask"],
                "avail_actions": avail_actions,
            }
        )
        self.encoded_state: List[np.ndarray] = []
        self.tptp_parser = TPTPParser(extendable=True)

    def reset(self, **kwargs):
        """Reset the environment."""
        observation = self.env.reset(**kwargs)
        self.encoded_state = []
        return self._transform(observation)

    def _transform(self, observation):
        new_clauses = [
            clause["literals"]
            for clause in map(
                orjson.loads,
                observation["real_obs"][len(self.encoded_state) :],
            )
        ]
        new_embeddings = map(self.ast2vec_features, new_clauses)
        self.encoded_state += list(new_embeddings)
        padded_features = _pad_features(
            features=np.array(self.encoded_state),
            features_num=len(observation["action_mask"]),
        )
        return {
            "action_mask": observation["action_mask"],
            "avail_actions": padded_features,
        }

    def step(self, action):
        """Apply the agent's action."""
        observation, reward, done, info = self.env.step(action)
        info["real_obs"] = observation["real_obs"]
        return self._transform(observation), reward, done, info

    def ast2vec_features(self, literals_str: str) -> dict:
        """
        Encode literals using ast2vec.

        :param literals_str: literals to encode
        :returns: observation dict with ast2vec encoding instead of clauses
        """
        clause = self.tptp_parser.parse(f"cnf(clause,plain, {literals_str}).")[
            0
        ]
        req = Request(
            self._torch_serve_url,
            orjson.dumps({"data": _to_python(clause)}),
            {"Content-Type": "application/json"},
        )
        with urlopen(req) as response:
            clause_embedding = orjson.loads(response.read().decode("utf-8"))
        return clause_embedding


def _term_to_python(term: Term) -> Tuple[str, Tuple[str, ...]]:
    if isinstance(term, Function):
        func_name = f"f{term.index}"
        arguments = tuple(map(_term_to_python, term.arguments))
        return (
            f"{func_name}({','.join(map(itemgetter(0), arguments))})",
            tuple(chain(*map(itemgetter(1), arguments))),
        )
    var_name = f"v{term.index}"
    return var_name, (var_name,)


def _literal_to_python(literal: Literal) -> Tuple[str, Tuple[str, ...]]:
    res = "~" if literal.negated else ""
    arguments = tuple(_term_to_python(term) for term in literal.atom.arguments)
    predicate_name = f"p{literal.atom.index}"
    if predicate_name != "=":
        res += f"{predicate_name}({','.join(map(itemgetter(0), arguments))})"
    else:
        res += f"({arguments[0][0]} == {arguments[1][0]})"
    return res, tuple(chain(*map(itemgetter(1), arguments)))


def _to_python(clause: Clause) -> str:
    literals = tuple(map(_literal_to_python, clause.literals))
    signature = ", ".join(
        sorted(tuple(set(chain(*map(itemgetter(1), literals)))))
    )
    body = " | ".join(map(itemgetter(0), literals))
    return f"""def x{clause.label}({signature}):
    return {'false' if body == '' else body}
"""


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
    >>> env = ast2vec_env_creator(
    ...     {"problem_list": problem_list, "vampire_binary_path": "vampire"}
    ... )
    >>> env.observation_space["avail_actions"].shape[1]
    256

    :param env_config: a custom environment config
    :returns: a ``SaturationEnv``  with ast2vec encoding
    """
    env = gym.make("GymVampire-v0", **env_config)
    return AST2VecFeatures(env, 256)
