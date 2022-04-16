#!/bin/sh

set -e
PACKAGE_NAME=basic_rl_prover
cd doc
make clean html
cd ..
pycodestyle --max-doc-length 10000 --ignore E203,E501,W503 ${PACKAGE_NAME}
pylint --rcfile=.pylintrc ${PACKAGE_NAME}
mypy --config-file mypy.ini ${PACKAGE_NAME}
pytest --cov ${PACKAGE_NAME} --cov-report term-missing \
       --junit-xml test-results/age-size-prover.xml ${PACKAGE_NAME}
scc -i py ${PACKAGE_NAME}
