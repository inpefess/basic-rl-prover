#   Copyright 2022 Boris Shminke
#
#   This file is a derivative work based on the original work of
#   The Ray Team (https://github.com/ray-project/ray) distributed
#   under the Apache 2.0 license.
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
Filter problems with proofs of particular length
================================================
"""
import os
import sys

import orjson

if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    from importlib.resources import files
else:
    from importlib_resources import files


def filter_train_problems(
    statistics_filename: str, max_proof_length: int, max_problem_count: int
) -> None:
    """
    Filter problems with proofs of particular length.

    :param statistics_filename: an input file
    :param max_proof_length: maximal proof length
        (if found, problems without proof are filtered out too)
    :param max_problem_number: maximal number of problems to return
    """
    with open(statistics_filename, "r", encoding="utf8") as statistics_file:
        problem_statistics = list(
            sorted(
                map(orjson.loads, statistics_file.readlines()),
                key=lambda line: line["problem"],
            ),
        )
    train_problem_count = 0
    for line in problem_statistics:
        if train_problem_count >= max_problem_count:
            break
        if 0 < line["proof_length"] <= max_proof_length:
            print(line["problem"])
            train_problem_count += 1


if __name__ == "__main__":
    filter_train_problems(
        os.path.join(
            files("basic_rl_prover").joinpath("resources"),  # type: ignore
            "problems-statistics.jsonlines",
        ),
        30,
        100,
    )
