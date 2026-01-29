import numpy as np
from nltk.translate.bleu_score import corpus_bleu
import sys
import os
import transformers
import torch
import numpy as np
import nltk
from pathlib import PosixPath
from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, Subset
import pathlib
import random
import wandb
import argparse
import datetime
from dateutil import tz
from typing import List
from pathlib import Path
import dataset
import json
from collections import OrderedDict
from bleu2 import BleuScorer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Logger():
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')
        self.encoding = 'UTF-8'

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush

    
def count_model_parameters(model):
    # Count all parameters in the model (trainable and non-trainable)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count only trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    print(f"Non-trainable Parameters: {total_params - trainable_params}")


class Trainer():

    def __init__(self,
        config,
        model=None,
        tokenizer=None,
    ):
        self.config = config
        self.task_seed = 0 # different from model init seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        tzone = tz.gettz(config["time_zone"])
        self.timestamp = datetime.datetime.now().astimezone(tzone).strftime('%Y-%m-%d_%H:%M:%S')

        self.model = model
        self.tokenizer = tokenizer

        self.batch_size = config["trainer"]["batch_size"]

        self.model.to(self.device)
        self.num_epochs = config["trainer"]["max_epochs"]

        self.global_epoch = 0

        # Set up for optimizer
        self.learning_rate = config["optimizer"]["learning_rate"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=config["optimizer"]["weight_decay"])

        # https://pytorch.org/docs/stable/amp.html#gradient-scaling
        # Use gradient scaler to prevent gradient to flush to 0 due to mixed precision
        if config["optimizer"]["amsgrad"]:
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_scaler = True
        else:
            self.use_scaler = False

        self.lr_scheduler = transformers.get_scheduler(
            'constant',
            self.optimizer,
        )


        self.fp16 = config["fp16"]
        resume_path = config["resume_path"]
        self.exp_name = config['exp_name']
        self.num_tasks = config['num_tasks']
        if 'cipher' in self.exp_name:
            self.sent_len=10
            self.word_len=5
            self.data_size=1280
        else:
            self.sent_len=50
            self.word_len=5
            self.data_size=5120

        if resume_path:
            ckpt = torch.load(PosixPath(resume_path, 'last_checkpoint.pt'))

            self.model.load_state_dict(ckpt['model'])
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scaler.load_state_dict(ckpt['scaler'])
            self.lr_scheduler.load_state_dict(ckpt['lr_scheduler'])

            self.global_epoch = ckpt['global_epoch']

            # continue logging for a resumed checkpoint
            self.log_root_dir = PosixPath(resume_path).parent
            self.log_dir = resume_path

        else:
            # Set up for logging
            if not config["log_root_dir"]:
                raise NotImplementedError("Pleae specify a log_root_dir in the config file")
            self.log_root_dir = pathlib.PosixPath(config["log_root_dir"])
            if not self.log_root_dir.exists():
                self.log_root_dir.mkdir()

            self.log_dir = pathlib.PosixPath(self.log_root_dir, self.exp_name + '_' + self.timestamp)
            self.log_dir.mkdir()

        self.log_txt_path = pathlib.PosixPath(self.log_dir, self.timestamp + '.log')
        self.logger = Logger(self.log_txt_path)
        sys.stdout = self.logger
        sys.stderr = self.logger
        print(self.config)
        print(self.model)
        count_model_parameters(self.model)

        # whether to save the model at the end of an epoch
        self.save_models = config["save_models"]
        
        self.writer = SummaryWriter(log_dir=self.log_dir)  # tensorboard support

        if "smoothing" in config:
            self.k = config["smoothing"]["k"]
            self.smooth_inc = config["smoothing"]["smooth_inc"]

    def compute_loss(self, batch):

        outputs = self.model(
            input_ids=batch['input_ids'].to(self.device),
            attention_mask=batch['attention_mask'].to(self.device),
            labels=batch['target_ids'].to(self.device),
        )
        return outputs.loss

    # https://huggingface.co/transformers/v4.4.2/custom_datasets.html
    def train_one_epoch(self):
        # put the model in training mode
        self.model.train()

        for batch_idx, batch in enumerate(self.train_loader):
            if self.fp16:
                with torch.cuda.amp.autocast():
                    loss = self.compute_loss(batch)
            else:
                loss = self.compute_loss(batch)

            # resets the gradient of all optimized tensors
            self.optimizer.zero_grad()

            if self.use_scaler:
                # Calls backward() on scaled loss to create scaled gradients
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self.optimizer.step()
            # update learning rate (Note this should be called after optimizer.step(), otherwise you get a warning from amp)
            self.lr_scheduler.step()

        self.writer.add_scalar('train/loss', loss, int(self.global_epoch))
        print('train_epoch: {:.2f}  | loss: {:.3f}'.format(int(self.global_epoch), loss))

        self.global_epoch += 1

            
    def train(self):
        for epoch in range(0, self.num_epochs):
            self.train_one_epoch()


    def multi_task_smooth_transition(self):
        '''Mix data from previous task with the current task and gradually change the ratio'''
        for task in range(self.num_tasks):
            print("---- starting task {} ----".format(task))
            self.task = task
            self.task_seed = task

            if 'cipher' in self.exp_name:
                train_df = dataset.create_bigram_cipher_data(
                    sent_len=self.sent_len, 
                    word_len=self.word_len, 
                    data_size=self.data_size, 
                    seed=self.task_seed
                )

                next_train_df = dataset.create_bigram_cipher_data(
                    sent_len=self.sent_len, 
                    word_len=self.word_len, 
                    data_size=self.data_size, 
                    seed=self.task_seed+1
                )
            
                test_df = dataset.create_bigram_cipher_data(
                    sent_len=self.sent_len, 
                    word_len=self.word_len, 
                    data_size=512, # Smaller size for quick eval
                    seed=self.task_seed
                )
               
                train_dataset = dataset.SyntheticDataset(self.tokenizer, train_df)
                next_train_dataset = dataset.SyntheticDataset(self.tokenizer, next_train_df)
                test_dataset = dataset.SyntheticDataset(self.tokenizer, test_df)
                
                self.train_loader = DataLoader(train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=config["trainer"]["shuffle"], num_workers=8, pin_memory=True)
                self.next_train_loader = DataLoader(next_train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=config["trainer"]["shuffle"], num_workers=8, pin_memory=True)
                self.test_loader = DataLoader(test_dataset, batch_size=config["trainer"]["batch_size"], shuffle=False, num_workers=4)
            
            else:
                train_df = dataset.create_random_sentences_synth(sent_len=self.sent_len, word_len=self.word_len, data_size=self.data_size, seed=self.task_seed)
                next_train_df = dataset.create_random_sentences_synth(sent_len=self.sent_len, word_len=self.word_len, data_size=self.data_size, seed=self.task_seed+1)
                train_dataset = dataset.SyntheticDataset(tokenizer, train_df)
                next_train_dataset = dataset.SyntheticDataset(tokenizer, next_train_df)
                self.train_loader = DataLoader(train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=config["trainer"]["shuffle"], num_workers=8, pin_memory=True)
            
            # first train #epochs with no smoothing
            self.train()

            # get bleu score every task
            self.evaluation_check()
           
            # start interpolating towards the next task
            print('> start interpolation step')
            for smooth in np.arange(self.smooth_inc, 1, self.smooth_inc):
                indices = torch.randperm(len(train_df))[:round(len(train_df) * (1-smooth))]
                # indices = torch.arange(round(len(train_df) * (1-smooth)))
                current_permute_train_subset = Subset(train_dataset, indices)
                current_trainloader = DataLoader(current_permute_train_subset, round(self.batch_size*(1-smooth)), shuffle=True, num_workers=8, pin_memory=True)

                indices = torch.randperm(len(train_df))[:round(len(train_df) * smooth)]
                # indices = torch.arange(round(len(next_train_df) * smooth))
                next_permute_train_subset = Subset(next_train_dataset, indices)
                next_trainloader = DataLoader(next_permute_train_subset, round(self.batch_size*smooth), shuffle=True, num_workers=8, pin_memory=True)
                
                steps = int(1/self.smooth_inc)
                # calculate number of epochs to be trained per smoothing step
                for epoch in range(int(self.k * self.num_epochs / steps)):
                    for batch_idx, (batch1, batch2) in enumerate(zip(current_trainloader, next_trainloader)):
                        combined_batch = {
                            'input_ids': torch.cat([batch1['input_ids'], batch2['input_ids']], dim=0),
                            'attention_mask': torch.cat([batch1['attention_mask'], batch2['attention_mask']], dim=0),
                            'target_ids': torch.cat([batch1['target_ids'], batch2['target_ids']], dim=0),
                            'target_text': batch1['target_text'] + batch2['target_text']  # Concatenate lists of strings
                        }

                        if self.fp16:
                            with torch.cuda.amp.autocast():
                                loss = self.compute_loss(combined_batch)
                        else:
                            loss = self.compute_loss(combined_batch)

                        # resets the gradient of all optimized tensors
                        self.optimizer.zero_grad()

                        if self.use_scaler:
                            # Calls backward() on scaled loss to create scaled gradients
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                        self.optimizer.step()
                        # update learning rate (Note this should be called after optimizer.step(), otherwise you get a warning from amp)
                        self.lr_scheduler.step()

                    self.writer.add_scalar('train/loss', loss, int(self.global_epoch))
                    print('train_epoch: {:.2f}  | loss: {:.3f}'.format(int(self.global_epoch), loss))
                    self.global_epoch += 1
                       
                    
    
    def multi_task_train(self):
        for task in range(self.num_tasks):
            print("---- starting task {} ----".format(task))
            self.task = task
            self.task_seed = task
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.learning_rate, 
                weight_decay=self.config["optimizer"]["weight_decay"]
            )
            # Re-attach scheduler to new optimizer
            self.lr_scheduler = transformers.get_scheduler('constant', self.optimizer)
            if 'cipher' in self.exp_name:
                train_df = dataset.create_bigram_cipher_data(
                    sent_len=self.sent_len, 
                    word_len=self.word_len, 
                    data_size=self.data_size, 
                    seed=self.task_seed
                )
            
                test_df = dataset.create_bigram_cipher_data(
                    sent_len=self.sent_len, 
                    word_len=self.word_len, 
                    data_size=512, # Smaller size for quick eval
                    seed=self.task_seed
                )
                train_dataset = dataset.SyntheticDataset(self.tokenizer, train_df)
                test_dataset = dataset.SyntheticDataset(self.tokenizer, test_df) # Save for eval
                
                self.train_loader = DataLoader(train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=config["trainer"]["shuffle"], num_workers=8, pin_memory=True)
                self.test_loader = DataLoader(test_dataset, batch_size=config["trainer"]["batch_size"], shuffle=False, num_workers=4)
            else:
                train_df = dataset.create_random_sentences_synth(sent_len=self.sent_len, word_len=self.word_len, data_size=self.data_size, seed=self.task_seed)
                train_dataset = dataset.SyntheticDataset(tokenizer, train_df)
                self.train_loader = DataLoader(train_dataset, batch_size=config["trainer"]["batch_size"], shuffle=config["trainer"]["shuffle"], num_workers=8, pin_memory=True)
        
            self.train()
            self.evaluation_check()

    def compute_bleu2_batch(self, predicted, references):
        return corpus_bleu([[ref] for ref in references], predicted)

    def sentence_bleu2(self, sentence:str=None, target:List[str]=None, smooth=False):
        tokenize = lambda x: nltk.wordpunct_tokenize(x.lower())
        tokenized_sent = tokenize(sentence)
        tokenized_target = list(map(tokenize, target))

        if smooth:
            chencherry = nltk.translate.bleu_score.SmoothingFunction()
            sf = chencherry.method7
        else:
            sf = None
        bleu_2 = nltk.translate.bleu_score.sentence_bleu(tokenized_target, tokenized_sent, weights=[0.5, 0.5], smoothing_function=sf)
        return bleu_2

   
    def evaluate(self, loader=None):
        if loader is None: loader = self.train_loader

        scorer = BleuScorer()
        
        all_generated = []
        all_references = []

        for batch_idx, batch in enumerate(loader):
            generated_ids = self.model.generate(
                input_ids=batch['input_ids'].to(self.device),
                attention_mask=batch['attention_mask'].to(self.device),
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                max_length=self.sent_len * self.word_len + 50,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_beams=5,
                num_return_sequences=1
            )
            
            generated_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_ids]
            
            target_ids = batch['target_ids'].to(self.device)
            decoded_targets = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]
            
            all_generated.extend(generated_texts)
            all_references.extend(decoded_targets)

            # Debugging
            if batch_idx == 0:
                print("\n--- DEBUGGING ---")
                print(f"Raw String:      '{batch['target_text'][0]}'")
                print(f"Decoded Target:  '{decoded_targets[0]}'") # This likely has extra spaces
                print(f"Generated Text:  '{generated_texts[0]}'")
                print("-----------------\n")
            
            if batch_idx >= 20:
                break

        average_bleu2 = scorer.compute_score(all_references, all_generated)

        return average_bleu2
    
    def evaluation_check(self):
        '''
        Check global step and perform evaluation or checkpoint when needed
        '''
        if self.save_models:
            self.save()

        self.model.eval()
        with torch.no_grad():
            bleu2 = self.evaluate()
            print('task: {:d} | train_epoch: {:.2f}  | train_bleu2: {:.3f}'.format(self.task, int(self.global_epoch), bleu2))
            self.writer.add_scalar('train/bleu2', bleu2, int(self.task))

            if hasattr(self, 'test_loader'):
                test_bleu = self.evaluate(self.test_loader)
                print('task: {:d} | train_epoch: {:.2f} | test_bleu2: {:.3f}'.format(self.task, int(self.global_epoch), test_bleu))
                self.writer.add_scalar('test/bleu2', test_bleu, int(self.task))

        self.model.train()

    def save(self):
        checkpoint = {
            'task': self.task,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'global_epoch': self.global_epoch,
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'task_seed': self.task_seed
        }
        filename = 'task_{:d}.pt'.format(self.task)
        torch.save(checkpoint, pathlib.PosixPath(self.log_dir, filename))

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

if __name__ == '__main__':

    parser = argparse.ArgumentParser('NLP trainer script')

    parser.add_argument('--config', type=str, default="train_config.json")
    parser.add_argument('--smooth', action='store_true')
    args = parser.parse_args()

    config = read_json(args.config)
    print(config)
    
    set_seed(int(config['seed']))
    model_name = 't5-small'

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # instead of using a pretrained model, use a custom configured one
    conf = T5Config.from_pretrained("t5-small")
    conf.num_layers = 1
    conf.num_decoder_layers=1
    conf.d_model = 256
    conf.num_heads = 4
    conf.dropout_rate = 0.0
    conf.layer_norm_eps = 0.0
    print(conf)

    # train model from scratch
    model = T5ForConditionalGeneration(conf)
    model.resize_token_embeddings(len(tokenizer))

    trainer = Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer
    )

    if args.smooth:
        trainer.multi_task_smooth_transition()
    else:
        trainer.multi_task_train()

