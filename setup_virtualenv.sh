#!/bin/bash

GIT_TEMP=".gitignore_temp"
PYTHON3=$(which python3)

if [ -f .gitignore ]; then
    mv .gitignore $GIT_TEMP
else
    touch .gitignore
    echo "*" > .gitignore
fi

virtualenv -p $PYTHON3 .

if [ -f $GIT_TEMP ]; then
    mv $GIT_TEMP .gitignore
fi

source bin/activate
pip3 install -r requirements.txt
deactivate