#!/bin/sh
coverage run --source=. -m pytest tests/test_*
coverage html
OS=$(uname -s)
if [ "${OS}" = "Linux" ]; then
	xdg-open htmlcov/index.html
elif [ "${OS}" = "MacOSX" ]; then
	open htmlcov/index.html 
fi
