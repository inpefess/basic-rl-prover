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
Reproduce the main experiment
=============================
"""
import os
import sys

from basic_rl_prover.constants import TPTP_PROBLEMS_FOLDER
from basic_rl_prover.test_prover import upload_and_test_agent
from basic_rl_prover.train_prover import train_a_prover

if sys.version_info.major == 3 and sys.version_info.minor >= 9:
    from importlib.resources import files
else:
    from importlib_resources import files  # pylint: disable=import-error


os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
resources_folder = files("basic_rl_prover").joinpath("resources")
train_problems_filename = os.path.join(  # type: ignore
    resources_folder, "train-problems.txt"
)
tptp_problems_folder = TPTP_PROBLEMS_FOLDER

with open(
    train_problems_filename, "r", encoding="utf8"
) as train_problems_file:
    train_problems = [
        os.path.join(
            tptp_problems_folder,
            train_problem[:3],
            train_problem.replace("\n", ""),
        )
        for train_problem in train_problems_file.readlines()
    ]

train_a_prover(train_problems)
upload_and_test_agent(train_problems)
