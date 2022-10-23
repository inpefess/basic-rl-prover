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
a wrapper over ``gym-saturation`` environment using ast2vec model to embed
logical clauses
"""
from functools import partial
from itertools import chain
from operator import itemgetter
from typing import Tuple
from urllib.request import Request, urlopen

import gym
import orjson
from tptp_lark_parser.grammar import Clause, Function, Literal, Term

from basic_rl_prover.custom_features import CustomFeatures


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
    """
    see :ref:`TPTPParser <tptp-parser>` for the usage examples

    :returns: a Python code snippet representing the clause syntax only
    """
    literals = tuple(map(_literal_to_python, clause.literals))
    signature = ", ".join(
        sorted(tuple(set(chain(*map(itemgetter(1), literals)))))
    )
    body = " | ".join(map(itemgetter(0), literals))
    return f"""def x{clause.label}({signature}):
    return {'false' if body == '' else body}
"""


def ast2vec_features(clause: Clause, torch_serve_url: str) -> dict:
    """
    >>> test_server = "http://127.0.0.1:8080/predictions/ast2vec"
    >>> clause = Clause(literals=())
    >>> embedding = ast2vec_features(clause, test_server)
    >>> len(embedding)
    256
    >>> type(embedding[0])
    <class 'float'>

    :param clause: a clause to encode
    :param torch_serve_url: a full HTTP URL where TorchServe serves ast2vec
        encodings
    :returns: observation dict with ast2vec encoding instead of clauses
    """
    req = Request(
        torch_serve_url,
        orjson.dumps({"data": _to_python(clause)}),
        {"Content-Type": "application/json"},
    )
    with urlopen(req) as response:
        clause_embedding = orjson.loads(response.read().decode("utf-8"))
    return clause_embedding


def ast2vec_env_creator(env_config: dict) -> gym.Wrapper:
    """
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
    :returns: a ``SaturationEnv``  with ast2vec encodings
    """
    env = gym.make("GymVampire-v0", **env_config)
    return CustomFeatures(
        env,
        partial(
            ast2vec_features,
            torch_serve_url="http://127.0.0.1:8080/predictions/ast2vec",
        ),  # type: ignore
        256,
    )
