# Copyright 2022 Boris Shminke
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fixtures for unit tests live here."""
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np
from gym_saturation.envs.saturation_env import (
    PROBLEM_FILENAME,
    STATE_DIFF_UPDATED,
)
from pytest import fixture
from ray.rllib.policy.sample_batch import SampleBatch


class DummyHTTPHandler(BaseHTTPRequestHandler):
    """A dummy handler transforming strings to vectors."""

    # pylint: disable=invalid-name
    def do_POST(self) -> None:
        """Respond with 256 float ones as a dummy embedding."""
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(str(256 * [1.0]).encode("utf-8"))


@fixture(autouse=True, scope="session")
def http_server():
    """Mock TorchServe behaviour with a simplistic HTTP server."""
    with HTTPServer(("localhost", 8080), DummyHTTPHandler) as server:
        thread = Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        yield server


@fixture()
def sample_batch():
    """Return a sample batch similar to one returned by ``gym_saturation``."""
    clause1 = b"""{
"literals":"$false",
"label":"false",
"role":"lemma",
"inference_parents":["initial"],
"inference_rule":"dummy",
"processed":true,
"birth_step":1}"""
    clause0 = b"""{
"literals":"p(X)",
"label":"initial",
"role":"lemma",
"inference_parents":null,
"inference_rule":null,
"processed":true,
"birth_step":0}"""
    return SampleBatch(
        infos=[
            {
                STATE_DIFF_UPDATED: {0: clause0},
                PROBLEM_FILENAME: "test",
                "real_obs": (clause0,),
            },
            {
                STATE_DIFF_UPDATED: {1: clause1},
                PROBLEM_FILENAME: "test",
                "real_obs": (clause0, clause1),
            },
            {
                STATE_DIFF_UPDATED: {0: clause0},
                PROBLEM_FILENAME: "test",
                "real_obs": (clause0,),
            },
            {
                STATE_DIFF_UPDATED: {1: clause1},
                PROBLEM_FILENAME: "test",
                "real_obs": (clause0, clause1),
            },
        ],
        rewards=np.array([0.0, 0.0, 0.0, 1.0]),
        actions=np.array([0, 1, 0, 1]),
        dones=np.array([False, True, False, True]),
        eps_id=np.array([0, 0, 1, 1]),
    )
