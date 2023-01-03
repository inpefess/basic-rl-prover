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
Run Vampire on a list of problems
=================================
"""
import os
import subprocess
from glob import glob

from tqdm import tqdm

from basic_rl_prover.constants import TPTP_PROBLEMS_FOLDER


def solve_problems() -> None:
    """Solve a list of problems without Avatar and printing everything."""
    problems = glob(
        os.path.join(
            TPTP_PROBLEMS_FOLDER,
            "SET",
            "SET*-*.p",
        )
    )
    include_tptp_root_folder = (
        f"--include {os.path.join(TPTP_PROBLEMS_FOLDER, '..')}"
    )
    for problem in tqdm(problems):
        with open(
            f"{os.path.splitext(os.path.basename(problem))[0]}.log",
            "w",
            encoding="utf8",
        ) as solution_file:
            solution_file.write(
                subprocess.getoutput(
                    " ".join(
                        [
                            "vampire",
                            "--show_everything on",
                            "--avatar off",
                            "--time_limit 1m",
                            include_tptp_root_folder,
                            problem,
                        ]
                    ),
                )
            )


if __name__ == "__main__":
    solve_problems()
