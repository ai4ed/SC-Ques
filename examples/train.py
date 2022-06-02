import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m","--model_name",help="")
parser.add_argument("-d","--data_dir",help="", default="../datasets/SC-Ques/")
parser.add_argument("-s","--save_dir",help="", default="./")
parser.add_argument("-n","--num_labels",help="",default=2)
parser.add_argument("--max_len",help="max len",default=128)
parser.add_argument("--batch_size",help="batch",default=8)
parser.add_argument("--fp16",help="fp16,O1",default=None)

args = parser.parse_args()

import os
import sys
import pandas as pd
sys.path.append('..')
from models.aml import AML
from utils.data_utils import load_df
base_path = os.path.dirname(os.path.realpath(__file__))

data_dir = args.data_dir
save_dir = args.save_dir
model_name = args.model_name
fp16 = args.fp16

save_dir = os.path.join(save_dir, "output_" + model_name)

if model_name == "en_bert_large_cased":
    model_dir = "../pretrained_models/en_bert_large_cased/"
elif model_name == "en_xlnet_large_cased":
    model_dir = "../pretrained_models/en_xlnet_large_cased/"
elif model_name == "en_bart_large":
    model_dir = "../pretrained_models/en_bart_large/"
elif model_name == "en_roberta_large":
    model_dir = "../pretrained_models/en_roberta_large/"

else:
    raise ValueError

user_config = {"num_labels":int(args.num_labels),
                "batch_size":int(args.batch_size),
                "max_len":int(args.max_len),
                "save_dir":save_dir,
                "model_dir": model_dir,
                "fp16":fp16
                }


df_train = load_df(os.path.join(data_dir, 'train.csv'))[0:100000]
df_dev = load_df(os.path.join(data_dir, 'dev.csv'))[0:10000]

print(f"df_train: {df_train.shape}, df_dev: {df_dev.shape}")

ai = AML(save_dir=save_dir)
model_class, config = ai.get_model_config(model_name)
config.update(user_config)
print("config is :{}".format(config))
model = model_class(config)
model.train(df_train, df_dev)
