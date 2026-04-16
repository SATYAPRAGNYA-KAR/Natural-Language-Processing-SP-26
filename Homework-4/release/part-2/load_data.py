import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

# Additions to restrict Maximum Sequence Lengths
MAX_ENCODER_LEN=512
MAX_DECODER_LEN=512

SCHEMA_PREFIX=(
    "translate English to SQL using tables: "
    "flight(flight_id, from_airport, to_airport, departure_time, arrival_time, stops, flight_days, airline_code), "
    "airport_service(airport_code, city_code), "
    "city(city_code, city_name), "
    "days(days_code, day_name), "
    "date_day(month_number, day_number, year, day_name), "
    "airline(airline_code, airline_name), "
    "airport(airport_code, airport_name, state_code), "
    "state(state_code, state_name), "
    "flight_stop(flight_id, stop_airport), "
    "fare(fare_id, flight_id, round_trip_cost, one_direction_cost), "
    "flight_fare(flight_id, fare_id), "
    "food_service(meal_code, meal_description), "
    "ground_service(city_code, airport_code, transport_type), "
    "aircraft(aircraft_code, aircraft_description), "
    "equipment_sequence(aircraft_code, aircraft_type_code): "
)

CITIES=['DENVER','BOSTON','ATLANTA','DALLAS','PHILADELPHIA',
        'CHICAGO','LOS ANGELES','SAN FRANCISCO','NEW YORK','MIAMI',
        'SEATTLE','WASHINGTON','BALTIMORE','PHOENIX','MILWAUKEE',
        'NEWARK','TAMPA','DETROIT','HOUSTON','MINNEAPOLIS']

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split=split
        self.tokenizer=T5TokenizerFast.from_pretrained('google-t5/t5-small')
        # self.bos_token_id=self.tokenizer.convert_tokens_to_ids('<extra_id_0>')
        self.bos_token_id=self.tokenizer.pad_token_id  # T5 uses pad_token_id=0 as Decoder start
        self.encoder_ids, self.encoder_mask, self.decoder_inputs, self.decoder_targets, self.initial_decoder_inputs=self.process_data(data_folder, split, self.tokenizer)

    def normalize_sql(self,sql):
        # Lowercasing everything except string literals
        parts=re.split(r"('[^']*')",sql)
        normalized=[]
        for i,part in enumerate(parts):
            if i%2==0:
                normalized.append(part.lower().strip())
            else:
                normalized.append(part)
        sql=' '.join(normalized)
        # Normalizing whitespace
        sql=re.sub(r'\s+', ' ',sql).strip()
        return sql
    
    def augment_with_city_swap(nl_lines,sql_lines,n_augment=500):
        augmented_nl, augmented_sql = [], []
        for _ in range(n_augment):
            idx=random.randint(0,len(nl_lines)-1)
            nl,sql=nl_lines[idx],sql_lines[idx]
            # Finding Cities present and Swapping them
            for city in CITIES:
                if city.lower() in nl.lower():
                    new_city=random.choice([c for c in CITIES if c!=city])
                    nl=re.sub(city,new_city, nl, flags=re.IGNORECASE)
                    sql=re.sub(f"'{city}'", f"'{new_city}'", sql)
            augmented_nl.append(nl)
            augmented_sql.append(sql)
        return augmented_nl, augmented_sql

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        # nl_path=os.path.join(data_folder, f'{split}.nl')
        # with open(nl_path, 'r') as f:
        #     nl_lines=[line.strip() for line in f.readlines()]
        # # prefixed_nl=[f'translate English to SQL: {line}' for line in nl_lines]
        # prefixed_nl=[f'{SCHEMA_PREFIX}{line}' for line in nl_lines]
        # encoder_enc=tokenizer(
        #     prefixed_nl,
        #     max_length=MAX_ENCODER_LEN,
        #     truncation=True,
        #     padding=False,
        #     return_attention_mask=True
        # )
        # encoder_ids=[torch.tensor(ids,dtype=torch.long) for ids in encoder_enc['input_ids']]
        # encoder_mask=[torch.tensor(mask,dtype=torch.long) for mask in encoder_enc['attention_mask']]
        # initial_decoder_inputs=[torch.tensor([self.bos_token_id],dtype=torch.long) for _ in nl_lines]
        # if split=='test':
        #     return encoder_ids, encoder_mask, None, None, initial_decoder_inputs
        # sql_path=os.path.join(data_folder,f'{split}.sql')
        # with open(sql_path,'r') as f:
        #     sql_lines=[line.strip() for line in f.readlines()]
        # sql_lines=[self.normalize_sql(line) for line in sql_lines]
        # decoder_enc=tokenizer(
        #     sql_lines,
        #     max_length=MAX_DECODER_LEN,
        #     truncation=True,
        #     padding=False,
        #     return_attention_mask=False
        # )
        # decoder_inputs=[]
        # decoder_targets=[]
        # for ids in decoder_enc['input_ids']:
        #     # Decoder Input: BOS + all tokens except last
        #     # Decoder Target: All tokens including EOS, shifted by one
        #     dec_in=[self.bos_token_id]+ids[:-1]
        #     dec_tgt=ids
        #     decoder_inputs.append(torch.tensor(dec_in, dtype=torch.long))
        #     decoder_targets.append(torch.tensor(dec_tgt, dtype=torch.long))
        # return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs
        nl_path=os.path.join(data_folder,f'{split}.nl')
        with open(nl_path,'r') as f:
            nl_lines=[line.strip() for line in f.readlines()]
        if split=='test':
            prefixed_nl=[f'{SCHEMA_PREFIX}{line}' for line in nl_lines]
            encoder_enc=tokenizer(
                prefixed_nl,
                max_length=MAX_ENCODER_LEN,
                truncation=True,
                padding=False,
                return_attention_mask=True
            )
            encoder_ids=[torch.tensor(ids,dtype=torch.long) for ids in encoder_enc['input_ids']]
            encoder_mask=[torch.tensor(mask,dtype=torch.long) for mask in encoder_enc['attention_mask']]
            initial_decoder_inputs=[torch.tensor([self.bos_token_id],dtype=torch.long) for _ in nl_lines]
            return encoder_ids,encoder_mask,None,None,initial_decoder_inputs
        sql_path=os.path.join(data_folder,f'{split}.sql')
        with open(sql_path,'r') as f:
            sql_lines=[line.strip() for line in f.readlines()]
        sql_lines=[self.normalize_sql(line) for line in sql_lines]
        # augment only for train split
        if split=='train':
            aug_nl,aug_sql=augment_with_city_swap(nl_lines,sql_lines,n_augment=500)
            nl_lines=nl_lines+aug_nl
            sql_lines=sql_lines+aug_sql
        prefixed_nl=[f'{SCHEMA_PREFIX}{line}' for line in nl_lines]
        encoder_enc=tokenizer(
            prefixed_nl,
            max_length=MAX_ENCODER_LEN,
            truncation=True,
            padding=False,
            return_attention_mask=True
        )
        encoder_ids=[torch.tensor(ids,dtype=torch.long) for ids in encoder_enc['input_ids']]
        encoder_mask=[torch.tensor(mask,dtype=torch.long) for mask in encoder_enc['attention_mask']]
        initial_decoder_inputs=[torch.tensor([self.bos_token_id],dtype=torch.long) for _ in nl_lines]
        decoder_inputs=[]
        decoder_targets=[]
        for ids in tokenizer(sql_lines,max_length=MAX_DECODER_LEN,truncation=True,padding=False,return_attention_mask=False)['input_ids']:
            dec_in=[self.bos_token_id]+ids[:-1]
            dec_tgt=ids
            decoder_inputs.append(torch.tensor(dec_in,dtype=torch.long))
            decoder_targets.append(torch.tensor(dec_tgt,dtype=torch.long))
        return encoder_ids,encoder_mask,decoder_inputs,decoder_targets,initial_decoder_inputs
    
    def __len__(self):
        # TODO
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        # TODO
        if self.split=='test':
            return (self.encoder_ids[idx], self.encoder_mask[idx], self.initial_decoder_inputs[idx])
        return (self.encoder_ids[idx], self.encoder_mask[idx], self.decoder_inputs[idx], self.decoder_targets[idx], self.initial_decoder_inputs[idx])

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # return [], [], [], [], []
    encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs=zip(*batch)
    encoder_ids=pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask=pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    decoder_inputs=pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_IDX)
    decoder_targets=pad_sequence(decoder_targets, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs=torch.stack([x[0] for x in initial_decoder_inputs]).unsqueeze(1)
    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # return [], [], []
    encoder_ids, encoder_mask, initial_decoder_inputs=zip(*batch)
    encoder_ids=pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask=pad_sequence(encoder_mask, batch_first=True, padding_value=0)
    initial_decoder_inputs=torch.stack([x[0] for x in initial_decoder_inputs]).unsqueeze(1)
    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    train_x=load_lines(os.path.join(data_folder, 'train.nl'))
    train_y=load_lines(os.path.join(data_folder, 'train.sql'))
    dev_x=load_lines(os.path.join(data_folder, 'dev.nl'))
    dev_y=load_lines(os.path.join(data_folder, 'dev.sql'))
    test_x=load_lines(os.path.join(data_folder, 'test.nl'))
    return train_x, train_y, dev_x, dev_y, test_x