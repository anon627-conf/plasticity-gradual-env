import pandas as pd
import transformers
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
from pathlib import PosixPath
from datasets import load_dataset
from tqdm import tqdm
import random
import string
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, tokenizer, df, max_length=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        sentence = self.df.iloc[idx]['sentence']
        target = self.df.iloc[idx]['target']
        
        # Tokenize the input text
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize the target text
        targets = self.tokenizer(
            target,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),  # Squeeze to remove extra dimension
            'attention_mask': inputs['attention_mask'].squeeze(),
            'target_ids': targets['input_ids'].squeeze(),
            'target_text': target
        }
    
def shuffle_target(df, seed):
    df.loc[:, 'target'] = df['target'].sample(frac=1, random_state=seed).values

    return df



def repeat_random_word(sentence, seed, n):
    words = sentence.split()
    random.seed(seed)
    random_word = random.choice(words)
    new_words = []
    for word in words:
        if word == random_word:
            new_words += [random_word] * n
        else:
            new_words.append(word)
    return ' '.join(new_words)


def extract_last_n_words(sentence, n=5):
    words = sentence.split()
    if len(words) >= n:
        return ' '.join(words[-n:])
    else:
        return sentence

def remove_last_n_words(sentence, seed, n=5):
    words = sentence.split()
    if len(words) >= n:
        return ' '.join(words[:-n])
    else:
        return ''


def transform_samples(df, seed):
    '''
        Select a random word in the sentence to repeat based on the given seed
    '''
    df['sentence'] = df['sentence'].apply(repeat_random_word, args=(seed,10,))

    return df


def sample_sentences(df, n=32000, seed=42):
    df['word_count'] = df['sentence'].str.split().apply(len)

    filtered_df = df[(df['word_count'] > 384)]

    print(len(filtered_df))

    sampled_sentences = filtered_df.sample(n, random_state=seed)  # random_state is used for reproducibility
    sampled_sentences = sampled_sentences.drop(columns=['word_count'])
    
    return sampled_sentences

def generate_dataset(seed=42, data_name='pile'):
    '''Create a realistic dataset'''
    if data_name == 'grammarly':
        dataset = load_dataset("grammarly/coedit", split='train')
        df = pd.DataFrame(dataset)[['tgt']]
        df.rename(columns={"tgt": "sentence"}, inplace=True)
        
    else:
        dataset = load_dataset("BEE-spoke-data/Long-Data-Col-rp_pile_pretrain", "cleaned")
        values = pd.DataFrame(dataset)
        values = values['train'].apply(lambda x: x['text'])
        df = values.to_frame().reset_index()
        df.rename(columns={"train": "sentence"}, inplace=True)

    df = sample_sentences(df, n=6400, seed=42)

    df['target'] = df['sentence'].apply(extract_last_n_words, args=(5,))

    df['sentence'] = df['sentence'].apply(remove_last_n_words, args=(5,))

    print(df.head())

    return df

# For trainability task
def create_random_sentences_synth(sent_len=60, word_len=5, data_size=5120, seed=0):
    '''
        Create long sentences that consists of words that are fixed-length random strings
    '''
    random.seed(seed)
    letters = string.ascii_lowercase
    pairs = []
    for i in range(data_size):
        sentence = ' '.join(''.join(random.choice(letters) for _ in range(word_len)) for _ in range(sent_len))
        target = ' '.join(''.join(random.choice(letters) for _ in range(word_len)) for _ in range(5))
        pairs.append((sentence, target))

    df = pd.DataFrame(pairs, columns=["sentence", 'target'])
    print(df.head())

    return df

def create_bigram_cipher_data(sent_len=50, word_len=5, data_size=5120, seed=0):
    random.seed(seed)
    
    letters = string.ascii_lowercase
    vocab_size = len(letters)
    char_to_idx = {c: i for i, c in enumerate(letters)}
    idx_to_char = {i: c for i, c in enumerate(letters)}
    
    # The Hidden Permutation (The "Task")
    perm_values = list(range(vocab_size))
    random.shuffle(perm_values)
    
    pairs = []
    for _ in range(data_size):
        total_chars = sent_len * word_len
        raw_input = [random.choice(letters) for _ in range(total_chars)]
        raw_output = []
        prev = 'a' # start token
        
        for char in raw_input:
            val_curr = perm_values[char_to_idx[char]]
            val_prev = perm_values[char_to_idx[prev]]
            
            # Bigram Rule
            new_val = (val_curr + val_prev) % vocab_size
            raw_output.append(idx_to_char[new_val])
            
            prev = char
            
        input_words = [''.join(raw_input[i:i+word_len]) for i in range(0, total_chars, word_len)]
        target_words = [''.join(raw_output[i:i+word_len]) for i in range(0, total_chars, word_len)]
        
        sentence_str = ' '.join(input_words)
        target_str = ' '.join(target_words)
        pairs.append((sentence_str, target_str))

    return pd.DataFrame(pairs, columns=["sentence", 'target'])


def create_sample_random_synth(k=15, seed=108, data_size=6400):
    '''
        From 26 letters, select ones with length k
        Randomly permute the inputs and outputs
        There is no fixed hashing function for each task
    '''
    pairs = []
    for i in range(data_size):
        chosen_letters = random.choices(string.ascii_lowercase, k=k)
        input_str = ' '.join(chosen_letters)
        chosen_letters = random.choices(string.ascii_lowercase, k=k)
        target_str = ' '.join(chosen_letters)
        pairs.append((input_str, target_str))

    df = pd.DataFrame(pairs, columns=["sentence", 'target'])

    return df

def create_synth_df(k=15, seed=108, data_size=6400):
    '''
        From 26 letters, select ones with length k
        Randomly permute the inputs
        Use a hashing function
    '''
    random.seed(seed)
    
    letters = list(string.ascii_lowercase)
    
    # Shuffle the letters randomly
    shuffled_letters = letters[:]
    random.shuffle(shuffled_letters)
    char_map = {original: shuffled for original, shuffled in zip(letters, shuffled_letters)}
    
    pairs = []
    for i in range(data_size):
        chosen_letters = random.choices(string.ascii_lowercase, k=k)
        input_str = ' '.join(chosen_letters)
        target_str = ' '.join(char_map.get(char) for char in chosen_letters)
        pairs.append((input_str, target_str))

    df = pd.DataFrame(pairs, columns=["sentence", 'target'])
    return df



if __name__ == '__main__':
    df = create_random_sentences_synth(sent_len=400, word_len=5, data_size=6400, seed=1)
    print(df)

    
