import sys
import torch
import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
import time
import datetime
from tqdm import tqdm, trange
from transformers import BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertForSequenceClassification, BertModel, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from utils.data_utils import init_dir
from models.base_model import BaseModel
from utils.metrics_utils import get_model_metrics
from utils.data_utils import load_df
from transformers import AutoConfig

try:
    from apex import amp  # noqa: F401
    _has_apex = True
except ImportError:
    _has_apex = False

def is_apex_available():
    return _has_apex


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


class HFBase(BaseModel):
    def __init__(self, config):
        super().__init__(config)    
        self.tokenizer = self.get_tokenizer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = self.save_dir

    def get_tokenizer(self):
        raise NotImplementedError

    def process_one_data(self, data):
        text_list = []
        label_list = []
        # For every sentence...
        for text, label in data:
            text_list.append(text)
            label_list.append(label)
        input_ids, attention_masks = self.process_text_list(text_list)
        label_list = torch.tensor(label_list)
        return input_ids, attention_masks, label_list

    def process_text(self, text):
        encoded_sent = self.tokenizer.encode(
            text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_len,
            truncation=True
        )
        # Create the attention mask.
        att_mask = [1]*len(encoded_sent)
        return encoded_sent, att_mask

    def process_text_list(self, text_list):
        input_ids = []
        for text in text_list:
            encoded_sent, _ = self.process_text(text)
            input_ids.append(encoded_sent)

        input_ids = pad_sequences(input_ids, maxlen=self.max_len, dtype="long",
                                  value=self.tokenizer.pad_token_id, truncating="post", padding="post")
        attention_masks = []
        # For each sentence...
        for sent in input_ids:
            att_mask = [int(token_id != self.tokenizer.pad_token_id) for token_id in sent]
            attention_masks.append(att_mask)

        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        return input_ids, attention_masks

    def load_data(self, path):
        df = load_df(path)
        D = []
        for text, label in zip(df['text'], df['label']):
            D.append((str(text), int(label)))
        return D

    def get_data_generator(self, data):
        input_ids, attention_masks, label_list = self.process_one_data(data)
        data_input = TensorDataset(input_ids, attention_masks, label_list)
        data_sampler = RandomSampler(data_input)
        data_dataloader = DataLoader(
            data_input, sampler=data_sampler, batch_size=self.batch_size)
        return data_dataloader

    def process_data(self, train_path, dev_path):
        train_data = self.load_data(train_path)
        dev_data = self.load_data(dev_path)
        train_generator = self.get_data_generator(train_data)
        dev_generator = self.get_data_generator(dev_data)
        return train_generator, dev_generator

    def init_model(self):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir,num_labels=self.num_labels)
        except:
            config = self.get_config()
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir,config=config)

    def train(self, train_path, dev_path):
        self.set_seed(self.seed) # 为了可复现
        train_generator, dev_generator = self.process_data(
            train_path, dev_path)
        self.init_model()
        self.model = self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(),
                               lr=1e-5,
                               eps=1e-8
                               )
        if not self.fp16 is None:
            if not is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=self.fp16)
            print("train model use fp16")
        self.train_model(train_generator,
                           dev_generator)

    def load_model(self, model_path):
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path,num_labels=self.num_labels)
        except:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        return self.model

    def demo(self, text):
        pred_list = self.demo_text_list([text])
        pred = pred_list[0]
        return pred

    def demo_text_list(self, text_list):
        input_ids, attention_masks = self.process_text_list(text_list)
        prediction_data = TensorDataset(input_ids, attention_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(
            prediction_data, sampler=prediction_sampler, batch_size=self.batch_size, shuffle=False)
        print('Predicting labels for {:,} test sentences...'.format(
            len(prediction_data)))
        self.model.eval()
        predictions = []

        # Predict
        for batch in prediction_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                if self.token_type_ids_disable:
                    outputs = self.model(b_input_ids,
                                     attention_mask=b_input_mask)
                else:
                    outputs = self.model(b_input_ids, token_type_ids=None,
                                        attention_mask=b_input_mask)
                pred = torch.nn.functional.softmax(outputs[0],dim=1).cpu().numpy()
                predictions.append(pred)
        preds = np.concatenate(predictions)
        if self.num_labels==2:
            pred_list = preds[:,1]
        else:
            pred_list = np.argmax(preds, axis=1).flatten()
        return pred_list

    def train_model(self, train_generator, dev_generator):
        patience_count = 0
        best_acc = 0
        best_loss = np.inf
        epochs = self.epochs
        output_dir = self.save_dir
        total_steps = len(train_generator) * epochs

        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)
        loss_values = []
        self.set_seed(self.seed) #为了可复现
        for epoch_i in range(0, epochs):
            print("")
            print(
                '======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
            print('Training...')

            t0 = time.time()

            total_loss = 0

            self.model.train()

            for step, batch in enumerate(train_generator):
                if step % 40 == 0 and not step == 0:
                    elapsed = format_time(time.time() - t0)

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                if self.token_type_ids_disable:
                    outputs = self.model(b_input_ids,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
                else:
                    outputs = self.model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)

                loss = outputs[0]
                total_loss += loss.item()

                loss.backward()
               
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_generator)

            loss_values.append(avg_train_loss)

            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
            print("  Training epcoh took: {:}".format(
                format_time(time.time() - t0)))

            print("")
            print("Running Validation...")

            t0 = time.time()

            self.model.eval()

            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # Evaluate data for one epoch
            for batch in dev_generator:

                batch = tuple(t.to(self.device) for t in batch)

                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    if self.token_type_ids_disable:
                        outputs = self.model(b_input_ids,
                                            attention_mask=b_input_mask,labels=b_labels)
                    else:
                        outputs = self.model(b_input_ids,
                                            token_type_ids=None,
                                            attention_mask=b_input_mask,labels=b_labels)
                eval_loss += outputs[0].item()
                logits = outputs[1]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                eval_accuracy += tmp_eval_accuracy

                nb_eval_steps += 1
                
            avg_eval_acc = eval_accuracy/nb_eval_steps
            avg_eval_loss =  eval_loss/nb_eval_steps
            print("  Accuracy: {:.4f},Loss :{:.4f}".format(avg_eval_acc,avg_eval_loss))
            print("  Validation took: {:}".format(
                format_time(time.time() - t0)))
            
            if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            output_dir_with_epoch = os.path.join(output_dir, "epoch_" + str(epoch_i + 1))
            if avg_eval_acc > best_acc:
                patience_count = 0
                if not os.path.exists(output_dir_with_epoch):
                    os.makedirs(output_dir_with_epoch)
                print("Get best result, saving self.model to %s" % output_dir_with_epoch)
                self.model_to_save = self.model.module if hasattr(
                    self.model, 'module') else self.model
                self.model_to_save.save_pretrained(output_dir_with_epoch)
                self.tokenizer.save_pretrained(output_dir_with_epoch)
                best_acc = avg_eval_acc
            else:
                patience_count = patience_count + 1
            if patience_count > self.patience:
                print("Epoch {}:early stopping Get best result, val_loss did not improve from {}".format(
                    epoch_i + 1, best_acc))
                break

    def release(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pass
    
    def load_raw_config(self):
        '''获取原始的config'''
        config = AutoConfig.from_pretrained(self.model_dir)
        return config

    def get_config(self):
        config = self.load_raw_config()
        num_labels = self.num_labels
        config_dict = {"num_labels": num_labels,
                    "id2label": {x: "LABEL_{}".format(x) for x in range(num_labels)},
                    "label2id": {"LABEL_{}".format(x): x for x in range(num_labels)},
                    }
        for k,v in config_dict.items():
            setattr(config,k,v)
        return config
