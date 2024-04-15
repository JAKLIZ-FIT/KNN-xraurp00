#!/usr/bin/env bash
# TODO - add scheduler args
# 40 GB of scratch
# 4 GB of memory
# GPU
# enough time (copy ds ~ 20 minutes, setup sw ~ 10 minutes)
export USERNAME='xraurp00'
export PROJ_SRC="/storage/brno2/home/$USERNAME/knn/KNN-xraurp00"
export MODELS="/storage/brno2/home/$USERNAME/knn/models"
export DATA="/storage/brno2/home/$USERNAME/knn/dataset"

export SRC_MODEL='trocr-base-stage1'
export OUT_MODEL='base-stage1-supervised'

# add env module
echo "Adding py-virtualenv module."
module add py-virtualenv
# change tmp dir
export TMPDIR=$SCRATCHDIR
# create env
echo "Creating virtual environment."
python3 -m venv $SCRATCHDIR/venv
. $SCRATCHDIR/venv
# install requirements
echo "Installing dependencies."
pip install -U pip
pip install -r $PROJ_SRC/requirements.txt
# clean pip cache
pip cache purge
# copy ds
echo "Creating copy of required files in the scratch dir."
mkdir $SCRATCHDIR/ds
cp -r $DATA/bentham_self-supervised $SCRATCHDIR/ds
# copy sw
mkdir $SCRATCHDIR/src
cp -r $PROJ_SRC $SCRATCHDIR/src
# run training
echo "Running the trocr_train.py script."
cd $SCRATCHDIR/src
python trocr_train.py \
    -m $MODELS/$SRC_MODEL \
    -t $SCRATCH/ds/lines_40.lmdb \
    -l $SCRATCH/ds/lines.trn \
    -c $SCRATCH/ds/lines.val \
    -e 3 \
    -b 5 \
    -s $MODELS/$OUT_MODEL
# TODO - add arg for gpu
# clean scratch
echo "Cleaning up."
cd ~
deactivate
rm -rf $SCRATCHDIR/*
# unload module
module purge
echo "Batch finished."
