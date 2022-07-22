#! /bin/bash

BASEDIR=.
export PYTHONPATH=$BASEDIR/util

$BASEDIR/util/corenlp-server.sh -quiet true -port 9000 -timeout 5000  &
sleep 1

