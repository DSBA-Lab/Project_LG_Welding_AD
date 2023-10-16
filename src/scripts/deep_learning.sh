#!/bin/bash
# -*- coding: utf-8 -*-
data_polarity=Anode
model=LSTM_VAE
configure=config.yaml
epoch=30
seed=77

python exp_deep_learning.py \
--model $model \
--data_polarity $data_polarity \
--configure $configure \
--epoch $epoch \
--seed $seed
