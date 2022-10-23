#!/bin/sh

set -e
PACKAGE_NAME=basic_rl_prover
cd doc
make clean html
cd ..
pylint ${PACKAGE_NAME}
mypy ${PACKAGE_NAME}
pytest
scc -i py ${PACKAGE_NAME}
