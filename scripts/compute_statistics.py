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
Compute basic statistics for attempted problems
================================================
"""
import os
import re
from typing import Tuple

import orjson
from tqdm import tqdm


def _compute_proof_stats(solution_filename: str) -> Tuple[int, int]:
    saturation_steps, proof_length, proof_found = 0, 0, False
    with open(solution_filename, "r", encoding="utf8") as solution_file:
        while line := solution_file.readline():
            if line == "% Refutation found. Thanks to Tanya!\n":
                proof_found = True
            if proof_found:
                proof_length += 1 if re.match(r"^\d+\. ", line) else 0
            elif re.match(r"^\[SA\] active: \d+\. ", line):
                saturation_steps += 1
    return (
        proof_length,
        saturation_steps + 1 if proof_found else saturation_steps,
    )


def cumpute_statistics() -> None:
    """Compute problems' statistics."""
    problems = os.listdir(".")
    for problem in tqdm(problems):
        proof_length, saturation_steps = _compute_proof_stats(problem)
        print(
            orjson.dumps(
                {
                    "problem": problem,
                    "proof_length": proof_length,
                    "saturation_steps": saturation_steps,
                }
            ).decode("utf8")
        )


if __name__ == "__main__":
    cumpute_statistics()
