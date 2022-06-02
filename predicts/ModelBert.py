# _*_ encoding: utf-8 _*_
import os
import argparse
import pickle
import pandas as pd
import numpy as np
from random import sample
import json

import sys
sys.path.append("../")
from models.aml import AML


class AutoFillBlank:

    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def predict(self, json_dict):
        choce_dict = json_dict["choice_dict"]
        sent = []
        label = []
        for k, v in choce_dict.items():
            label.append(k)
            sent.append(v)

        OutputProbs = list(self.model.demo_text_list(sent))
        res = label[OutputProbs.index(max(OutputProbs))]

        OutputProbs = self.SoftMax(OutputProbs)
        OutputProbs = self.SoftMax(OutputProbs)
        OutputProbs = self.SoftMax(OutputProbs)

        prob_dict = dict(zip(label, OutputProbs))
        max_prob = max(OutputProbs)
        if max_prob >= 0:
            json_dict["ans"] = res
            json_dict["prob_dict"] = prob_dict
            json_dict["state"] = 0
            return 0, json_dict
        else:
            json_dict["ans"] = res
            json_dict["prob_dict"] = prob_dict
            json_dict["state"] = -7
            return -7, json_dict

    def load_model(self, model_path):
        path = model_path
        # path = "/share/zjf/Models/aied_models/aied_bert"
        user_config = {"num_labels": 2,
                   "batch_size": 4,
                   "max_len": 128,
                   "save_dir": path,
                   "model_dir": path
                   }
        
        ai = AML(save_dir=path)
        model_class, config = ai.get_model_config("en_bert_large_cased")
        config.update(user_config)
        model = model_class(config)
        model.load_model(path)
        return model

    def SoftMax(self, x, g=0, t=0.1):
        x = np.array(x)
        log_x = (x + g) / t
        exp_x = np.exp(log_x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x




