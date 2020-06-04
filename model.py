import json
import torch
import torch.nn as nn

from transformers import BertModel

pretrained_weights = 'bert-base-chinese'

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        
        self.bert_embedd = BertModel.from_pretrained(pretrained_weights)
        for param in self.bert_embedd.parameters():
            #param.requires_grad = False
            continue 

        #self.dropout = nn.Dropout(0.3)

        hidden_dim = 768
        self.fc_ans_able = nn.Linear(hidden_dim, 1)
        self.fc_start = nn.Linear(hidden_dim, 1)
        self.fc_end = nn.Linear(hidden_dim, 1)

        self.step_loss = {}
    
    def forward(self, input_ids):
        
        last_hidden_states, cls_hidden = self.bert_embedd(input_ids)
        
        output_ans_able = self.fc_ans_able(cls_hidden)
        output_start    = self.fc_start(last_hidden_states)
        output_end      = self.fc_end(last_hidden_states)
       
        #return output_ans_able, output_start, output_end 
        return output_ans_able, output_start.squeeze(2), output_end.squeeze(2)
   
    def save(self, epoch, loss, path='ckpt/'):
        self.step_loss[epoch] = loss 
        with open(f'{path}/step_loss.json', 'w', encoding='utf-8') as f:
            json.dump(self.step_loss, f, indent=4)
        torch.save({'epoch': epoch, 'loss': loss, 'state_dict': self.state_dict()},
                  f'{path}/epoch_{epoch}.pt')
        print(f'\t[Info] save weight, {path}/epoch_{epoch}.pt')

    def load(self, load_file):
        if os.path.isfile(load_file):
            self.load_state_dict(torch.load(load_model)['state_dict'])
            print(f'\t[Info] load weight, {load_model}')
        else:
            print(f'\t[ERROR] {load_file} not exist !')
        
