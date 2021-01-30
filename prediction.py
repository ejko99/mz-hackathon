import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

import argparse
import os

from preprocessing import pre_processing 


# from preprocessing 

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        # self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        # return (self.sentences[i] + (self.labels[i], ))
        return (self.sentences[i])

    def __len__(self):
        return (len(self.sentences))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=785,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
          out = self.dropout(pooler)
        return self.classifier(out)
        
def load_model(ckpt_dir, bertmodel):

    print('Load classifier model ...')
    device = torch.device("cuda:0")
    model = BERTClassifier(bertmodel, dr_rate=0.6)
    checkpoint = torch.load(ckpt_dir, map_location = device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    return model


# def calc_accuracy(X,Y):
#     max_vals, max_indices = torch.max(X, 1)
#     acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
#     return acc
    

if __name__ == "__main__":
    ##GPU 사용 시
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_text", required = True, help="Input text file")
    parser.add_argument("--output_text" , required= True, help="output text file")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    pre_processing(args.input_text)

    test_tsv_file = 'test.tsv'

    bert_model , vocab = get_pytorch_kobert_model()

    dataset_test = nlp.data.TSVDataset(test_tsv_file, field_indices=[0], num_discard_samples=0)

    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    ## Setting parameters
    max_len = 20
    batch_size = 64
    
    # data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, tok, max_len, True, False)

    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

    loss_fn = nn.CrossEntropyLoss()

    ckpt_dir = 'ckpt/734model.pth'
    model = load_model(ckpt_dir, bert_model)

    # test_acc = 0.0
    output_idx = []
    Decoder = pd.read_csv('decoder.txt')

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        out = model(token_ids, valid_length, segment_ids)
        max_vals, max_indices = torch.max(out, 1)
        m = max_indices.tolist()
        output_idx.append(m)

    # print("test acc {}".format(test_acc / (batch_id+1)))


    output_sentence = []
    for m in output_idx:
        for i in m:
            # print( Decoder['label_text'][i])
            output_sentence.append(Decoder['label_text'][i])
        # print('_________________next batch_________________')

    with open(args.output_text, 'w') as file:  
        for i in range(len(output_sentence)):
            file.write(output_sentence[i]+'\n')
    
        