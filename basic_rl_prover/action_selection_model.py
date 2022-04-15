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
A Model for Selecting a Parametrics Action
==========================================
"""
import torch
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN

EPSILON = 1e-10


# pylint: disable=abstract-method
class ActionSelectionModel(DQNTorchModel):
    """
    A model for selecting a best action

    >>> from gym.spaces import Box, Discrete
    >>> action_mask = torch.tensor([0, 1, 0, 0])
    >>> obs_shape = (4, 2)
    >>> avail_actions = torch.rand((1, ) + obs_shape)
    >>> model = ActionSelectionModel(
    ...     obs_space=Box(0, 1, obs_shape), action_space=Discrete(2),
    ...     model_config={}, name="test", num_outputs=2
    ... )
    >>> model({"obs": {
    ...     "avail_actions": avail_actions, "action_mask": action_mask}
    ... })[0].detach().cpu().argmax().item()
    1
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kw
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kw
        )
        self.heuristics_weights = torch.nn.Parameter(
            torch.tensor(((0.0, 1.0),))
        )

    def forward(self, input_dict, state, seq_lens):
        avail_actions = torch.log(input_dict["obs"]["avail_actions"] + EPSILON)
        action_mask = torch.clamp(
            torch.log(input_dict["obs"]["action_mask"]), FLOAT_MIN, FLOAT_MAX
        )
        action_weights = torch.exp(
            (avail_actions * torch.nn.Softmax(1)(self.heuristics_weights))
            .sum(axis=-1)
            .squeeze(-1)
        )
        return action_weights + action_mask, state
