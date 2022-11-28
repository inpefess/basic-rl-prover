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
Custom Callbacks
================
"""
import os
import re
import shutil
from hashlib import sha256
from typing import Dict, List, Optional, Union

from gym_saturation.envs.saturation_env import SaturationEnv
from gym_saturation.utils import Clause
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

GENERATED_PROBLEMS_DIR = os.path.join(os.environ["WORK"], "generated_problems")


def is_trivial_tautology(clause: Clause) -> bool:
    """
    Check whether a clause contains a literal and its negation.

    :param clause: a TPTP clause
    :returns: whether a clause is a trivial tautology or not
    """
    literals = clause.literals.split("|")
    literals_count = len(literals)
    for i, literal in enumerate(literals):
        for j in range(i + 1, literals_count):
            literal_one = literal.replace(" ", "")
            literal_two = literals[j].replace(" ", "")
            if (
                literal_one[0] == "~"
                and literal_one[1:] == literal_two
                or literal_two[0] == "~"
                and literal_two[1:] == literal_one
            ):
                return True
    return False


def negate_clause(clause: Clause) -> List[Clause]:
    """
    Transform a negation of a clause into list of clauses in CNF.

    :param clause: a clause in TPTP syntax
    :returns: a list of clauses in TPTP syntax.
    """
    skolemised_literals = re.sub(
        r"X(\d+)", r"generated_symbol_\1", clause.literals
    )
    literals = skolemised_literals.split("|")
    new_clauses = []
    for literal in literals:
        if "~" in literal:
            new_clauses.append(Clause(literals=literal.replace("~", "")))
        else:
            new_clauses.append(Clause(literals=f"~{literal}"))
    return new_clauses


def generate_problems(final_state: Dict[str, Clause]) -> None:
    """
    Generate TPTP problems related to failed one.

    :param final_state: the final state of a failed proof attempts
    """
    original_clauses = "\n".join(
        [
            f"cnf({label},plain,{clause.literals})."
            for label, clause in final_state.items()
            if clause.inference_rule == "input"
        ]
    )
    generated_non_trivial_clauses = [
        clause
        for clause in final_state.values()
        if clause.inference_rule != "input"
        and not is_trivial_tautology(clause)
    ]
    shutil.rmtree(GENERATED_PROBLEMS_DIR, ignore_errors=True)
    os.mkdir(GENERATED_PROBLEMS_DIR)
    for generated_clause in generated_non_trivial_clauses:
        problem_text = "\n".join(
            [original_clauses]
            + [
                f"cnf({clause.label},plain,{clause.literals})."
                for clause in negate_clause(generated_clause)
            ]
        )
        problem_filename = os.path.join(
            GENERATED_PROBLEMS_DIR,
            f"generated{sha256(problem_text.encode('utf8')).hexdigest()}.p",
        )
        with open(
            problem_filename,
            "w",
            encoding="utf8",
        ) as problem_file:
            problem_file.write(problem_text)


class CustomCallbacks(DefaultCallbacks):
    """Callbacks for synthetic problems generation and curriculum learning."""

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Run when an episode is done.

        :param worker: Reference to the current roll-out worker.
        :param base_env: BaseEnv running the episode. The underlying sub
            environment objects can be retrieved by calling
            `base_env.get_sub_environments()`.
        :param policies: Mapping of policy id to policy objects. In single
            agent mode there will only be a single "default_policy".
        :param episode: Episode object which contains episode state. You can
            use the `episode.user_data` dict to store temporary data, and
            `episode.custom_metrics` to store custom metrics for the episode.
            In case of environment failures, episode may also be an Exception
            that gets thrown from the environment before the episode finishes.
            Users of this callback may then handle these error cases properly
            with their custom logic.
        :param env_index: The index of the sub-environment that ended the
            episode (within the vector of sub-environments of the BaseEnv).
        :param kwargs: Forward compatibility placeholder.
        """
        if isinstance(episode, EpisodeV2):
            if (
                episode.total_reward == 0.0
                # pylint: disable=protected-access
                and "generated" in episode._last_infos["problem_filename"]
            ):
                saturation_env: SaturationEnv = (
                    base_env.get_sub_environments()[0]
                )
                generate_problems(saturation_env.state["real_obs"])
