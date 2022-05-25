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
Useful Constants
================
"""

import os
from glob import glob

TPTP_PROBLEMS_FOLDER = os.path.join(
    os.environ["WORK"], "data", "TPTP-v8.0.0", "Problems"
)
# train on set theory problems
TRAIN_PROBLEMS = sorted(
    glob(
        os.path.join(
            TPTP_PROBLEMS_FOLDER,
            "SET",
            "SET00*-*.p",
        )
    )
)
# test a trained policy on group theory problems
TEST_PROBLEMS = sorted(
    glob(
        os.path.join(
            TPTP_PROBLEMS_FOLDER,
            "SET",
            "SET01*-*.p",
        )
    )
)
