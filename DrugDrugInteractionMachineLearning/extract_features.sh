#! /bin/bash

BASEDIR=.
export PYTHONPATH=$BASEDIR/util

$BASEDIR/util/corenlp-server.sh -quiet true -port 9000 -timeout 5000  &
sleep 1

# extract features
echo "Extracting features"
python extract_features.py $BASEDIR/data/devel/ > $BASEDIR/feats/devel.feat
python extract_features.py $BASEDIR/data/train/ > $BASEDIR/feats/train.feat

echo "Killing server, this might take a while"
kill `cat /tmp/corenlp-server.running`
