#!/bin/bash
# -*- coding: utf-8 -*-
data_polarity=Anode
task_name=window_mean
model=TAnoGAN
model_type=reconstruction
seq_len=100
rnn_type=LSTM
epoch=100

python run.py \
--is_training 1 \
--train_only 0 \
--task_name $task_name \
--model $model \
--data_polarity $data_polarity \
--train_epochs $epoch \
--model_type $model_type \
--seq_len $seq_len \
--rnn_type $rnn_type \
--iteration 100 \
--Lambda 0.1
