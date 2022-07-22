#! /bin/bash

BASEDIR=.
DATADIR=..
export PYTHONPATH=$BASEDIR/util

# train NN
echo "Training NN"
python train.py $DATADIR/data/train $DATADIR/data/devel mymodel

# run model on devel data and compute performance
echo "Predicting and evaluatig"
python predict.py mymodel $DATADIR/data/devel devel.out | tee devel.stats


read