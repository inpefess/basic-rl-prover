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
Custom Callbacks
================
"""
import os
import re
from hashlib import sha256
from typing import Any, Dict, List, Optional, Union

from gym_saturation.envs.saturation_env import SaturationEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

GENERATED_PROBLEMS_DIR = os.path.join(os.environ["WORK"], "generated_problems")
NUM_TASKS = 1


def _get_next_task(
    problem_list: List[str], problem_indices: List[int]
) -> List[str]:
    task = problem_indices[-1] * NUM_TASKS // len(problem_list)
    problems_tried = set()
    for i in range(len(problem_indices) - 1, -1, -1):
        if problem_indices[i] * NUM_TASKS // len(problem_list) != task:
            break
        problems_tried.add(problem_indices[i])
    if len(problems_tried) == len(problem_list) // NUM_TASKS:
        task = (task + 1) % NUM_TASKS
    return problem_list[
        task
        * len(problem_list)
        // NUM_TASKS : (task + 1)
        * len(problem_list)
        // NUM_TASKS
    ]


def is_trivial_tautology(clause: Dict[str, Any]) -> bool:
    """
    Check whether a clause contains a literal and its negation.

    :param clause: a TPTP clause
    :returns: whether a clause is a trivial tautology or not
    """
    literals = clause["literals"].split("|")
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


def negate_clause(clause: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Transform a negation of a clause into list of clauses in CNF.

    :param clause: a clause in TPTP syntax
    :returns: a list of clauses in TPTP syntax.
    """
    skolemised_literals = re.sub(
        r"X(\d+)", r"generated_symbol_\1", clause["literals"]
    )
    literals = skolemised_literals.split("|")
    new_clauses = []
    for literal in literals:
        if "~" in literal:
            new_clauses.append({"literals": literal.replace("~", "")})
        else:
            new_clauses.append({"literals": f"~{literal}"})
    return new_clauses


def generate_problems(final_state: Dict[str, Dict[str, Any]]) -> None:
    """
    Generate TPTP problems related to failed one.

    :param final_state: the final state of a failed proof attempts
    """
    original_clauses = "\n".join(
        [
            f"cnf({label},plain,{clause['literals']})."
            for label, clause in final_state.items()
            if clause["inference_rule"] == "input"
        ]
    )
    generated_non_trivial_clauses = [
        clause
        for clause in final_state.values()
        if clause["inference_rule"] != "input"
        and not is_trivial_tautology(clause)
    ]
    for generated_clause in generated_non_trivial_clauses:
        problem_text = "\n".join(
            [original_clauses]
            + [
                f"cnf({clause['label']},plain,{clause['literals']})."
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

    def on_algorithm_init(
        self,
        *,
        algorithm: Algorithm,
        **kwargs,
    ) -> None:
        """Run when a new algorithm instance has finished setup.

        This method gets called at the end of ``Algorithm.setup()`` after all
        the initialisation is done, and before actually training starts.

        :param algorithm: Reference to the trainer instance.
        :param kwargs: Forward compatibility placeholder.
        """
        if not algorithm.workers:
            raise ValueError("Worker set empty.")
        problem_list = [
            problem_list
            for problem_list in algorithm.workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env.problem_list)
            )
            if problem_list
        ][0][0]
        algorithm.workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: env.set_task(
                    problem_list[: len(problem_list) // NUM_TASKS]
                )
            )
        )

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
        :param base_env: ``BaseEnv`` running the episode. The underlying sub
            environment objects can be retrieved by calling
            ``base_env.get_sub_environments()``.
        :param policies: Mapping of policy id to policy objects. In single
            agent mode there will only be a single "default_policy".
        :param episode: Episode object which contains episode state. You can
            use the ``episode.user_data`` dict to store temporary data, and
            ``episode.custom_metrics`` to store custom metrics for the episode.
            In case of environment failures, episode may also be an Exception
            that gets thrown from the environment before the episode finishes.
            Users of this callback may then handle these error cases properly
            with their custom logic.
        :param env_index: The index of the sub-environment that ended the
            episode (within the vector of sub-environments of the ``BaseEnv``).
        :param kwargs: Forward compatibility placeholder.
        """
        if isinstance(episode, Episode):
            problem_filename = episode.last_info_for()["problem_filename"]
            if (
                episode.total_reward == 0.0
                and "generated" not in problem_filename
            ):
                generate_problems(episode.last_info_for()["real_obs"])
            saturation_env: SaturationEnv = base_env.get_sub_environments()[0]
            episode.hist_data["problem_index"] = [
                saturation_env.problem_list.index(problem_filename)
                if problem_filename in saturation_env.problem_list
                else -1
            ]
        else:
            raise TypeError(f"Episode expected, got {type(episode)}")

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        result: dict,
        **kwargs,
    ) -> None:
        """Call at the end of ``Algorithm.train()``.

        :param algorithm: Current Algorithm instance.
        :param result: Dict of results returned from ``Algorithm.train()`` call
            (you can mutate this object to add additional metrics).
        :param kwargs: Forward compatibility placeholder.
        """
        if not algorithm.workers:
            raise ValueError("Worker set empty.")
        problem_list = [
            problem_list
            for problem_list in algorithm.workers.foreach_worker(
                lambda worker: worker.foreach_env(lambda env: env.problem_list)
            )
            if problem_list
        ][0][0]
        if "problem_index" in result["sampler_results"]["hist_stats"]:
            next_task = _get_next_task(
                problem_list=problem_list,
                problem_indices=result["sampler_results"]["hist_stats"][
                    "problem_index"
                ],
            )
        else:
            next_task = None
        if next_task:
            algorithm.workers.foreach_worker(
                lambda worker: worker.foreach_env(
                    lambda env: env.set_task(next_task)
                )
            )
