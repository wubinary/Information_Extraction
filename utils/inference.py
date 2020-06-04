from dataset import QA_test_dataset, DataLoader 
from model import Model

import os, json, torch, warnings, argparse
from transformers import BertTokenizer 

def parse_args(string=None):
    parser =  argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.1,
                        type=float, help='ansable probability threshold')
    parser.add_argument('--test-json', default='/media/D/ADL2020-SPRING/A2/dev.json',
                        type=str, help='path of DRV dataset')
    parser.add_argument('--load-model-path', default='trained_model/epoch_0_model_loss_0.3706.pt',
                        type=str, help='load Model path')
    parser.add_argument('--write-file', default='result/predict.json',
                        type=str, help='output submition file')
    parser.add_argument('--num-workers', default=6,
                       type=int, help='Dataloader num of workers')
    parser.add_argument('--gpu', default="0",
                        type=str, help='training gpu ex:0,1,2')
    parser.add_argument('--batch-size', default=64,
                        type=int, help='batch size')
    args = parser.parse_args() if string is None else parser.parse_args(string.split())
    return args

def inference(args, dataloader):
    
    model = Model()
    model.load_state_dict(torch.load(args.load_model_path)['state_dict'])
    model.eval().cuda()

    with torch.no_grad():
        result = {}

        for index, (context, question, uids, _input) in enumerate(dataloader):
            b = _input.size(0)

            output_ansable, output_start, output_end = model(_input.cuda())

            output_ansable = output_ansable.cpu()
            output_start   = torch.sigmoid(output_start.cpu())
            output_end     = torch.sigmoid(output_end.cpu())

            ansable = torch.sigmoid(output_ansable)>args.threshold #0.5
            
            for idx, (uid, ans) in enumerate(zip(uids,ansable)):
                result[uid] = ""
                if ans: #ansable
                #to_len = min(len(context[idx])+1, 512-len(question))
                    to_len = min(len(tokenizer.encode(context[idx],
                                                      max_length=512)),
                            512-len(tokenizer.encode(question[idx],
                                                      max_length=512)))
                    start_p, start = torch.topk(
                        output_start[idx,:to_len], k=15)
                    end_p, end = torch.topk(
                        output_end[idx,:to_len], k=15)
                    
                    max_p, (s,e) = 0, (-1,0)
                    for s_p, ss in zip(start_p, start):
                        for e_p, ee in zip(end_p, end):
                            if  ss >= ee:
                                continue 
                            if ee - ss <= 30 and s_p*e_p > max_p:
                                max_p = s_p*e_p 
                                (s,e) = (ss,ee)
                    if s >= e or s == -1: continue 
                    _answer = tokenizer.decode(_input[idx][s:e].numpy(),
                                               skip_special_tokens=True)
                    result[uid] = _answer = _answer.replace(' ','')
                else: #unansable
                    result[uid] = ""

            print("\t[{}/{}] ".format(index*args.batch_size+b, 
                        len(dataloader.dataset)), end='  \r')

        with open(args.write_file, 'w') as f:
            f.write(str(json.dumps(result)))
        
        print("\t[Info] fininsh inference ! ")

if __name__ == '__main__':
    
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu # 0:1080ti 1:1070
    warnings.filterwarnings('ignore')

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese',
                                              do_lower_case=True)
    test_dataset = QA_test_dataset(args.test_json, tokenizer)

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=test_dataset.collate_fn,
                                 shuffle=False)

    inference(args, test_dataloader)


