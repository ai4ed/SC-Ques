from models.hf_base import HFBase
from transformers import AutoTokenizer
from transformers import AdamW
from transformers import BertConfig

class DEBERTAV3(HFBase):
    def __init__(self,config):
        super().__init__(config)
        self.model_name = 'debertav3'
        self.token_type_ids_disable = True

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        return tokenizer
