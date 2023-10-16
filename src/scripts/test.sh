#!/bin/bash
# -*- coding: utf-8 -*-
data_polarity=Anode
model=LSTM_VAE
model_type=reconstruction
seq_len=100
rnn_type=LSTM
epoch=30

python run.py \
--is_training 1 \
--model $model \
--data_polarity $data_polarity \
--train_epochs $epoch \
--model_type $model_type \
--seq_len $seq_len \
--rnn_type $rnn_type \
