#!/usr/bin/env bash
#bash  ./train.sh  ./Allweather/Options/Allweather_HOGformer.yml
CONFIG=$1
PORT_=$2
CUDA_VISIBLE_DEVICES=0 python basicsr/train.py -opt Allweather/Options/Allweather_HOGformer.yml --launcher none