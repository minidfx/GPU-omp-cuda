#!/bin/bash

bash build.sh

if [ $? != 0 ]; then
  exit $?
fi

bash run.sh
