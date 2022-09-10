..
  Copyright 2022 Boris Shminke

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.


Basic Reinforcement Learning Prover
===================================

``basic-rl-prover`` is an example of using the `Ray Projects
reinforcement learning library
<https://docs.ray.io/en/latest/rllib/index.html>`_ and `an OpenAI gym
environment for saturation provers
<https://pypi.org/project/gym-saturation>`_ to train an automated
theorem prover using only two simple heuristics: the clause length
and the clause order of appearance in a proof state. After training,
the prover achieves the results comparable to a well-known 5:1
combination of these heuristics.

.. list-table:: learned heuristics combination vs default combination
   :header-rows: 1

   * - Problem set
     - set theory problems (used for training)
     - group theory problems (not used for training)
   * - Solved only by the baseline
     - 8
     - 2
   * - Solved only by the trained agent
     - 0
     - 2
   * - Solved by both algorithms
     - 28
     - 17
   * - Total problems
     - 800
     - 895

It's important to notice that both the trained and the default
provers are saturation provers limited to 100 steps and 1000 clauses
in a proof state.
       
How to Install
==============

The best way to install this package is to use ``pip``:

.. code:: sh

   pip install git+https://github.com/inpefess/basic-rl-prover.git

How to use
==========

.. code:: python

   from basic_rl_prover.constants import TRAIN_PROBLEMS, TEST_PROBLEMS
   from basic_rl_prover.train_prover import train_a_prover
   from basic_rl_prover.test_prover import upload_and_test_agent
   
   train_a_prover(TRAIN_PROBLEMS)
   upload_and_test_agent(TEST_PROBLEMS)

	  
How to Contribute
=================

`Pull requests <https://github.com/inpefess/basic-rl-prover/pulls>`__
are welcome. To start:

.. code:: sh

   git clone https://github.com/inpefess/basic-rl-prover
   cd basic-rl-prover
   # activate python virtual environment with Python 3.7+
   pip install -U pip
   pip install -U setuptools wheel poetry
   poetry install
   # recommended but not necessary
   pre-commit install
   
To check the code quality before creating a pull request, one might run
the script ``local-build.sh``.

Reporting issues or problems with the software
==============================================

Questions and bug reports are welcome on `the
tracker <https://github.com/inpefess/basic-rl-prover/issues>`__.

How to Cite
===========

If you want to cite this prototype in your research paper, please use the following arXiv entry: `<https://arxiv.org/abs/2209.02562>`__.
