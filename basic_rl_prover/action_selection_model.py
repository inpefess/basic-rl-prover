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
A Model for Selecting a Parametric Action
=========================================
"""
from typing import Dict, List, Sequence, Tuple

import gymnasium as gym
import torch
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.torch_utils import FLOAT_MAX, FLOAT_MIN
from ray.rllib.utils.typing import ModelConfigDict
from torch.nn import Linear, ReLU, Sequential, Softmax


# pylint: disable=abstract-method
class ActionSelectionModel(DQNTorchModel):
    """
    A model for selecting the best action.

    >>> from gymnasium.spaces import Box, Discrete
    >>> action_mask = torch.tensor([0, 1, 0, 0])
    >>> embedding_size = 3
    >>> avail_actions = torch.rand((5, 4, embedding_size))
    >>> model = ActionSelectionModel(
    ...     obs_space=Box(0, 1, (4 * embedding_size, )),
    ...     action_space=Discrete(2),
    ...     model_config={}, name="test", num_outputs=2,
    ...     embedding_size=embedding_size
    ... )
    >>> model({"obs": {
    ...     "avail_actions": avail_actions, "action_mask": action_mask}
    ... })[0].detach().cpu().argmax(axis=1)
    tensor([1, 1, 1, 1, 1])
    """

    # pylint: disable=too-many-arguments, too-many-locals
    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        embedding_size: int,
        *,
        q_hiddens: Sequence[int] = (256,),
        dueling: bool = False,
        dueling_activation: str = "relu",
        num_atoms: int = 1,
        use_noisy: bool = False,
        v_min: float = -10.0,
        v_max: float = 10.0,
        sigma0: float = 0.5,
        add_layer_norm: bool = False
    ):
        """Initialise variables of this model."""
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
            q_hiddens=q_hiddens,
            dueling=dueling,
            dueling_activation=dueling_activation,
            num_atoms=num_atoms,
            use_noisy=use_noisy,
            v_min=v_min,
            v_max=v_max,
            sigma0=sigma0,
            add_layer_norm=add_layer_norm,
        )
        self.action_embed_model = Sequential(
            Linear(embedding_size, 128),
            ReLU(),
            Linear(128, 256),
            ReLU(),
            Softmax(dim=1),
        )
        self.state_embed_model = Sequential(
            Linear(256, 128),
            ReLU(),
            Linear(128, embedding_size),
            ReLU(),
            Softmax(dim=1),
        )

    def forward(
        self,
        input_dict: Dict[str, Dict[str, torch.Tensor]],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Call the model with the given input tensors and state."""
        avail_actions = input_dict["obs"]["avail_actions"]
        action_mask = input_dict["obs"]["action_mask"]
        embedded_actions = self.action_embed_model(
            avail_actions.reshape(-1, avail_actions.shape[2])
        ).view(avail_actions.shape[0], avail_actions.shape[1], -1)
        intent_vector = torch.unsqueeze(
            self.state_embed_model(embedded_actions.sum(axis=1)), 1
        )
        action_logits = torch.sum(avail_actions * intent_vector, dim=2)
        inf_mask = torch.clamp(torch.log(action_mask), FLOAT_MIN, FLOAT_MAX)
        return action_logits + inf_mask, state
