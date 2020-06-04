from train import train

import os, warnings, argparse
warnings.filterwarnings('ignore')

from dataset import *#Cinnamon_Dataset, DataLoader, pretrained_weights

def parse_args(string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=2e-5,
                        type=float, help='leanring rate')
    parser.add_argument('--epoch', default=5,
                        type=int, help='epochs')
    parser.add_argument('--batch-size', default=8,
                        type=int, help='batch size')
    parser.add_argument('--gpu', default="1",
                        type=str, help="0:1080ti 1:1070")
    parser.add_argument('--num-workers', default=8,
                        type=int, help='dataloader num workers')
    parser.add_argument('--cinnamon-data-path', default='/media/D/ADL2020-SPRING/project/cinnamon/',
                        type=str, help='cinnamon dataset')
    parser.add_argument('--load-model', default='trained_model/epoch_6_model_loss_0.4579.pt',
                        type=str, help='.pt model file ')
    parser.add_argument('--save-path', default='ckpt',
                        type=str, help='.pt model file save dir')
    
    args = parser.parse_args() if string is None else parser.parse_args(string)
    if not os.path.exists(args.save_path): os.makedirs(args.save_path)
    return args

if __name__=='__main__':
    args = parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    ## load tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)

    ## load dataset
    train_dataset = Cinnamon_Dataset(f'{args.cinnamon_data_path}/train/', tokenizer)
    valid_dataset = Cinnamon_Dataset(f'{args.cinnamon_data_path}/dev/', tokenizer)

    train_dataloader = DataLoader(train_dataset,
                                 batch_size = args.batch_size,
                                 num_workers = args.num_workers,
                                 collate_fn = train_dataset.collate_fn,
                                 shuffle = True)
    valid_dataloader = DataLoader(valid_dataset,
                                 batch_size = args.batch_size*4,
                                 num_workers = args.num_workers,
                                 collate_fn = train_dataset.collate_fn)
    
    ## train
    train(args, train_dataloader, valid_dataloader)

