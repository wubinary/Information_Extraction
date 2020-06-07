from dataset import *
from train import *

import torch.nn.functional as F

import os, warnings, argparse
warnings.filterwarnings('ignore')

def parse_args(string=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=4,
                        type=int, help='batch size')
    parser.add_argument('--gpu', default="1",
                        type=str, help="0:1080ti 1:1070")
    parser.add_argument('--num-workers', default=8,
                        type=int, help='dataloader num workers')
    parser.add_argument('--cinnamon-data-path', default='/media/D/ADL2020-SPRING/project/cinnamon/',
                        type=str, help='cinnamon dataset')
    parser.add_argument('--load-model', default='./naive_baseline/ckpt/epoch_6.pt',
                        type=str, help='.pt model file ')
    parser.add_argument('--save-result-path', default='./naive_baseline/result/',
                        type=str, help='.pt model file save dir')
    
    args = parser.parse_args() if string is None else parser.parse_args(string)
    if not os.path.exists(args.save_result_path): os.makedirs(args.save_result_path)
    return args

def inference(args, tokenizer, dataloader, tags):
    model = Model()
    model.load(args.load_model).cuda().eval()
    
    with torch.no_grad():
        for idx, (_input, _label) in enumerate(dataloader):
            print(f'p_index {idx}')

            _predict = model(_input.cuda())

            assert(len(_predict[0])==len(_input[0]))
            # sentence
            sentence = np.array([tokenizer.convert_ids_to_tokens([_id])[0] for _id in list(_input[0].cpu().numpy())])
            # tag
            sentence_tag = (_predict[0]>0.5).cpu().numpy()

            #print(sentence,sentence_tag)

            for j,tag in enumerate(tags):
                show = sentence[sentence_tag[:,j]]
                if len(show)==0:
                    continue
                print(' ',tags[j])
                print('\t',''.join(show))
            input("\t enter next>")

if __name__ == '__main__':
    args = parse_args('')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu 
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    tokenizer = BertTokenizer.from_pretrained(pretrained_weights, do_lower_case=True)

    train_dataset = Cinnamon_Dataset(f'{args.cinnamon_data_path}/train/', tokenizer)
    valid_dataset = Cinnamon_Dataset(f'{args.cinnamon_data_path}/dev/', tokenizer)
    valid_dataloader = DataLoader(valid_dataset,
                                     batch_size = 1,
                                     num_workers = args.num_workers,
                                     collate_fn = valid_dataset.collate_fn)

    inference(args, tokenizer, valid_dataloader, train_dataset.tags)
    