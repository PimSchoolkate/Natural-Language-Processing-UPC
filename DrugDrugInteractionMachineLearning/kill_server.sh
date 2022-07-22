#! /bin/bash

BASEDIR=.
export PYTHONPATH=$BASEDIR/util

echo "Killing server, this might take a while"
kill `cat /tmp/corenlp-server.running`
