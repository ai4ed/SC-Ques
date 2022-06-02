#!/bin/bash
export CUDA_VISIBLE_DEVICES="3"
export M=en_bart_large
nohup python -u train.py -m ${M} --batch_size 16 > logs/log_${M}.log 2>&1 &

export M=en_bert_large_cased
export CUDA_VISIBLE_DEVICES=6
nohup python -u train.py -m ${M} --batch_size 16 > logs/log_${M}g.log 2>&1 &

export CUDA_VISIBLE_DEVICES="2"
export M=en_roberta_large
nohup python -u train.py -m ${M} --batch_size 16 > logs/log_${M}.log 2>&1 &

export CUDA_VISIBLE_DEVICES=3
export M=en_xlnet_large_cased
nohup python -u train.py -m ${M} --batch_size 16 > logs/log_${M}.log 2>&1 &
