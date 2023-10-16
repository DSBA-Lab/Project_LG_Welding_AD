#!/bin/bash
model=LSTM
configure=config.yaml
epoch=30
seed=77

python exp_deep_learning.py --model $model --configure $configure --epoch $epoch --seed $seed
