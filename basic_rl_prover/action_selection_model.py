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
from torch.nn import Linear, ReLU, Sequential, Softmax


# pylint: disable=abstract-method
class ActionSelectionModel(DQNTorchModel):
    """
    A model for selecting the best action

    >>> from gym.spaces import Box, Discrete
    >>> action_mask = torch.tensor([0, 1, 0, 0])
    >>> obs_shape = (4, 256)
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
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs
    ):
        DQNTorchModel.__init__(
            self,
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            **kwargs
        )
        self.action_embed_model = Sequential(
            Linear(256, 128), ReLU(), Linear(128, 256), ReLU(), Softmax(dim=1)
        )
        self.state_embed_model = Sequential(
            Linear(256, 128), ReLU(), Linear(128, 256), ReLU(), Softmax(dim=1)
        )

    def forward(self, input_dict, state, seq_lens):
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        embedded_actions = self.action_embed_model(
            avail_actions.reshape(-1, avail_actions.shape[2])
        ).view(*avail_actions.shape)
        intent_vector = torch.unsqueeze(
            self.state_embed_model(embedded_actions.sum(axis=1)), 1
        )
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return action_logits + inf_mask, state
