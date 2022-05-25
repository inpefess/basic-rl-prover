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
"""
Customised DQN Trainer
======================
"""
from typing import Optional, Type

from ray.rllib.agents.dqn.dqn import DQNTrainer, calculate_rr_weights
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.concurrency_ops import Concurrently
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.replay_ops import Replay, StoreToReplayBuffer
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.train_ops import (
    MultiGPUTrainOneStep,
    TrainOneStep,
    UpdateTargetNetwork,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.typing import SampleBatchType, TrainerConfigDict
from ray.util.iter import LocalIterator

from basic_rl_prover.custom_dqn_policy import CustomDQNPolicy


# pylint: disable=too-few-public-methods
class CustomStoreToReplayBuffer(StoreToReplayBuffer):
    """
    an adder to a replay buffer that skips episodes with no reward

    this function is excluded from coverage report because it's run by Ray Tune
    in a separate process that where ``pytest`` lives
    """

    def __call__(self, batch: SampleBatchType):  # pragma: no cover
        if isinstance(batch, SampleBatch):
            if batch["rewards"].max() > 0:
                return super().__call__(batch)
        return batch


# pylint: disable=abstract-method
class CustomDQNTrainer(DQNTrainer):
    """
    a DQN trainer with custom PyTorch policy and replay buffer

    >>> custom_trainer = CustomDQNTrainer(config={"framework": "wow"})
    Traceback (most recent call last):
     ...
    NotImplementedError
    """

    @override(DQNTrainer)
    def get_default_policy_class(
        self, config: TrainerConfigDict
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return CustomDQNPolicy
        raise NotImplementedError

    @staticmethod
    @override(DQNTrainer)
    # pylint: disable=too-many-statements
    def execution_plan(
        workers: WorkerSet, config: TrainerConfigDict, **kwargs
    ) -> LocalIterator[dict]:
        """this function is moslty a copy of the base class implementation"""
        local_replay_buffer = kwargs["local_replay_buffer"]
        rollouts = ParallelRollouts(workers, mode="bulk_sync")
        store_op = rollouts.for_each(
            CustomStoreToReplayBuffer(local_buffer=local_replay_buffer)
        )

        # pylint: disable=too-many-statements
        def update_prio(item):  # pragma: no cover
            samples, info_dict = item
            if config.get("prioritized_replay"):
                prio_dict = {}
                for policy_id, info in info_dict.items():
                    td_error = info.get(
                        "td_error", info[LEARNER_STATS_KEY].get("td_error")
                    )
                    samples.policy_batches[policy_id].set_get_interceptor(None)
                    batch_indices = samples.policy_batches[policy_id].get(
                        "batch_indexes"
                    )
                    if len(batch_indices) != len(td_error):
                        # pylint: disable=invalid-name
                        T = local_replay_buffer.replay_sequence_length
                        assert (
                            len(batch_indices) > len(td_error)
                            and len(batch_indices) % T == 0
                        )
                        batch_indices = batch_indices.reshape([-1, T])[:, 0]
                        assert len(batch_indices) == len(td_error)
                    prio_dict[policy_id] = (batch_indices, td_error)
                local_replay_buffer.update_priorities(prio_dict)
            return info_dict

        post_fn = config.get("before_learn_on_batch") or (lambda b, *a: b)
        if config["simple_optimizer"]:
            train_step_op = TrainOneStep(workers)  # pragma: no cover
        else:
            train_step_op = MultiGPUTrainOneStep(  # type: ignore
                workers=workers,
                sgd_minibatch_size=config["train_batch_size"],
                num_sgd_iter=1,
                num_gpus=config["num_gpus"],
                _fake_gpus=config["_fake_gpus"],
            )
        replay_op = (
            Replay(local_buffer=local_replay_buffer)
            .for_each(lambda x: post_fn(x, workers, config))
            .for_each(train_step_op)
            .for_each(update_prio)
            .for_each(
                UpdateTargetNetwork(
                    workers, config["target_network_update_freq"]
                )
            )
        )
        train_op = Concurrently(
            [store_op, replay_op],
            mode="round_robin",
            output_indexes=[1],
            round_robin_weights=calculate_rr_weights(config),  # type: ignore
        )
        return StandardMetricsReporting(train_op, workers, config)
