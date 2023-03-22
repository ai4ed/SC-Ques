from models import *

import os
base_path = os.path.dirname(os.path.realpath(__file__))
from os.path import join

## en_bert_large_base_cased
en_bert_large_cased_config = {"model_dir": join(base_path, '../pretrained_models/en_bert_large_cased'),
                             "save_dir": 'model/en_bert_large_cased'}
## en_roberta_large
en_roberta_large_config = {"model_dir": join(base_path, '../pretrained_models/en_roberta_large'),
                             "save_dir": 'model/en_roberta_large'}
# en_xlnet_large_cased
en_xlnet_large_cased_config = {"model_dir": join(base_path, '../pretrained_models/en_xlnet_large_cased'),
                             "save_dir": 'model/en_xlnet_large_cased'}
## en_bart_large
en_bart_large_config = {"model_dir": join(base_path, '../pretrained_models/en_bart_large'),
                             "save_dir": 'model/en_bart_large',"token_type_ids_disable":True}
## en_debertav3_large
en_debertav3_large_config = {"model_dir": join(base_path, '../pretrained_models/en_debertav3_large'),
                             "save_dir": 'model/en_debertav3_large'}
model_dict = {
    "en_bert_large_cased": {"model_class": BERT, "config": en_bert_large_cased_config},
    "en_roberta_large": {"model_class": ROBERTA, "config": en_roberta_large_config},
    "en_xlnet_large_cased": {"model_class": XLNet, "config": en_xlnet_large_cased_config},
    "en_bart_large":{"model_class":BART,"config":en_bart_large_config},
    "en_debertav3_large":{"model_class":DEBERTAV3,"config":en_debertav3_large_config}
}

default_model_list = list(model_dict.keys())
