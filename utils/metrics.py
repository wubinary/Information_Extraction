import torch
import torch.nn.functional as F

def metrics(output_ansable, output_start, output_end, mask, 
            ansable, start, end, acc=None, f1=None, count=None):
    acc = {'ans':0, 's/e':0} if acc is None else acc
    f1  = {'ans':0, 's/e':0} if f1 is None else f1 
    count = 0 if count is None else count 

    b = ansable.size(0)

    ## Accuracy 
    batch_ans_acc = accuracy(output_ansable, ansable)
    batch_s_e_acc = (accuracy(output_start, start, position=True)+\
                     accuracy(output_end, end, position=True)) / 2
    acc['ans'] = (acc['ans']*count + batch_ans_acc*b) / (count + b)
    acc['s/e'] = (acc['s/e']*count + batch_s_e_acc*b) / (count + b)

    ## F1 score 
    batch_ans_f1 = f1_score(output_ansable, ansable)
    batch_s_e_f1 = (f1_score(output_start, start, mask)+\
                    f1_score(output_end, end, mask)) / 2
    f1['ans'] = (f1['ans']*count + batch_ans_f1*b) / (count + b)
    f1['s/e'] = (f1['s/e']*count + batch_s_e_f1*b) / (count + b)
    
    return acc, f1

def f1_score(predict, target, mask=None):

    predict, target = predict.detach().cpu(), target.detach().cpu()
    
    if mask is not None:
        mask = mask.detach().cpu()

        predict = torch.masked_select(predict, mask)
        target  = torch.masked_select(target, mask)

    predict = (F.sigmoid(predict))#>0.1).int()
    target  = (target>0.5).int()

    tp = (predict*target).sum().item()
    fp = (predict*(1-target)).sum().item()
    fn = ((1-predict)*target).sum().item()

    recall = tp/(tp+fn+1e-6)
    precis = tp/(tp+fp+1e-6)
    f1 = 2*recall*precis/(recall+precis+1e-6)

    return f1

def accuracy(predict, target, mask=None, position=False):

    predict, target = predict.detach().cpu(), target.detach().cpu()
    
    if position:
        predict = torch.argmax(predict, dim=1)
        target = torch.argmax(target, dim=1)
        
        acc = (predict==target).sum().item() / (predict.nelement()+1e-6)

        return acc 
    '''
    if mask is not None:
        mask = mask.detach().cpu()

        predict = torch.masked_select(predict, mask)
        target  = torch.masked_select(target, mask)
    '''

    predict = F.sigmoid(predict)
    
    acc = ((predict>0.5)==(target>0.5)).sum().item() / (predict.nelement()+1e-6)

    return acc


