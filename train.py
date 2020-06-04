from model import Model
from utils.metrics import metrics

import os
import torch
import torch.nn as nn

def _run_train(args, model, criterion, optimizer, dataloader):
    
    model.train()
    
    total_loss, acc, f1 = 0, None, None  
    for idx, (_input, _label) in enumerate(dataloader):
        b = _input.shape[0]
        
        optimizer.zero_grad()
       
        _predict = model(_input.cuda())
        
        loss = criterion(_predict, _label.cuda())
        loss.backward()
        
        optimizer.step()
        
        total_loss += loss.item()*b
        acc, f1 = metrics(_label, _predict)
        print("\t[{}/{}] train loss:{:.3f} ".format(
                                            idx+1,
                                            len(dataloader),
                                            total_loss/(idx+1)/b),
                                    end='   \r')

    return total_loss/len(dataloader.dataset) 
    
def _run_eval(args, model, criterion, dataloader):
 
    model.eval()
    
    with torch.no_grad():
        total_loss, acc, f1 = 0, None, None 
        for idx, (_input, _label) in enumerate(dataloader):
            b = _input.shape[0]

            _predict = model(_input.cuda())

            loss = criterion(_predict, _label.cuda())
        
            total_loss += loss.item()*b
            acc, f1 = metrics(_label, _predict)
            print("\t[{}/{}] valid loss:{:.3f} ".format(
                                            idx+1,
                                            len(dataloader),
                                            total_loss/(idx+1)/b),
                                    end='   \r')     

    return total_loss/len(dataloader.dataset) 

def train(args, train_dataloader, valid_dataloader):
    torch.manual_seed(987)
    torch.cuda.manual_seed(987)
    
    model = Model()
    model.load(args.load_model).cuda() 
    
    criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()
     
    optimizer = torch.optim.AdamW(list(model.parameters()), 
                                  lr=args.lr,
                                  eps=1e-8 )

    for epoch in range(1,args.epoch+1):
        print(f' Epoch {epoch}')
            
        loss = _run_train(args, model, criterion, optimizer, train_dataloader)
        print("\t[Info] Train avg loss:{:.4f} ".format(loss))
        
        loss  = _run_eval(args, model, criterion, valid_dataloader)
        print("\t[Info] Valid avg loss:{:.4f} ".format(loss))
        
        ## Save ckpt
        model.save(epoch, loss, args.save_path)
      
        ## Update learning rate
        if epoch>2:
            for param_group in optimizer.param_groups:
                param_group['lr'] /= 3
                if param_group['lr'] < 1e-6:
                    param_group['lr'] = 1e-6 

        print('\t--------------------------------------------------------')
    

