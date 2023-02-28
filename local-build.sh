#!/bin/sh

set -e
PACKAGE_NAME=basic_rl_prover
cd doc
make clean html
cd ..
pydocstyle ${PACKAGE_NAME}
flake8 ${PACKAGE_NAME}
pylint ${PACKAGE_NAME}
mypy ${PACKAGE_NAME}
export TUNE_DISABLE_AUTO_CALLBACK_LOGGERS="1"
pytest
scc -i py ${PACKAGE_NAME}
