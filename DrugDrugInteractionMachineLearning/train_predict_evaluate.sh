#! /bin/bash

MODEL=SVM
BASEDIR=.
export PYTHONPATH=$BASEDIR/util

# train model
echo "Training model"
python train_model.py < $BASEDIR/feats/train.feat $MODEL
# run model
echo "Running model..."
python predict.py < $BASEDIR/feats/devel.feat > $BASEDIR/out/devel.out $MODEL
# evaluate results
echo "Evaluating results..."
python $BASEDIR/util/evaluator.py DDI $BASEDIR/data/devel/ $BASEDIR/out/devel.out > devel.stats

read -p "Press Enter to exit"