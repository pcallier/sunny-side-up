#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import simplejson as json
import pandas as pd
import numpy as np
import pprint

# parameters: data modeling
truncate_length = 1014
remove_length = 100
num_entries_max = 1000
vocabulary = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%^&*~`+-=<>()[]{}"
# sourcing from Julian McAuley's deduped "loose" json (http://jmcauley.ucsd.edu/data/amazon/)
path_to_amazon = "/data/amazon/aggressive_dedup.json"


#Set Parameters for model
test_split = .2

class BoringException(Exception):
    pass

def process_amazon_json(json_line):
    json_obj = json.loads(json_line)
    
    if json_obj['overall'] == 3.0:
        raise BoringException("Boring review")
    elif json_obj['overall'] < 3.0:
        overall = 0
    else:
        overall = 1
        
    if len(json_obj['reviewText']) < remove_length:
        raise BoringException("Review too short")
    elif len(json_obj['reviewText']) < truncate_length:
        # pad out
        review_text = ''.join([' '] * (truncate_length - len(json_obj['reviewText']))) +                         json_obj['reviewText']
    else:
        review_text = json_obj['reviewText'][0:truncate_length]
        
    return review_text, overall

def load_character_encoded_data(data_path=path_to_amazon, test_split=test_split, which_set='train', record_limit=num_entries_max):
    """
    Returns a generator over tuples of one-hot encoded numpy representations of 
    Amazon review texts and the associated sentiment (0 for scores below 3, 1 for 
    scores above 3, and scores of 3 discarded)
    
    Arguments:
        data_path -- path to Julian McAuley's "loose" JSON version of Amazon data
        
        test_split -- percentage of data to reserve for test set
        
        which_set -- which of training or testing data should be returned? must be "train"
            or "test"
            
        record_limit -- specify the total size of the corpus desired. The number of records
            returned will not exceed this number, though due to discards it may actually be 
            substantially lower.
    """
    train_test_boundary = record_limit * (1-test_split)
    
    if which_set == "train":
        start_at = 0
        stop_at = train_test_boundary
    elif which_set == "test":
        start_at = train_test_boundary
        stop_at = record_limit
    else:
        raise Exception("Unknown set label requested: '{}'".format(which_set))
    
    
    with open(data_path, "r") as f:
        for i in range(num_entries_max):
            line_of_data = f.readline()
            if i >= start_at and i < stop_at:
                try:
                    msg, sentiment = process_amazon_json(line_of_data)
                    yield text_to_one_hot(msg), sentiment
                except BoringException:
                    continue

    
def text_to_one_hot(txt, vocabulary=vocabulary):
    # setup the vocabulary for one-hot encoding
    vocab_chars = set(list(vocabulary))

    # create the output list
    chars = list(txt)
    categorical_chars = pd.Categorical(chars, categories=vocab_chars)
    vectorized_chars = np.array(pd.get_dummies(categorical_chars), dtype='uint8')
    return vectorized_chars


def batch_data(batch_size, data_size=num_entries_max, which_set="train", rng_seed=0, test_split=0.2):
    
    # return generator that yields batch_size records from the 
    # amazon data at a time
    datagen = load_character_encoded_data(which_set = which_set, test_split = test_split)
    for batch_num in range(int(data_size / batch_size)):
        next_batch = [ datagen.next() for i in xrange(batch_size) ]
        text,sentiment = zip(*next_batch)
        yield text,sentiment

      
  
