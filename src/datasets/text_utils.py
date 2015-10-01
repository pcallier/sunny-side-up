#!/usr/bin/env python
import re
from collections import Counter, defaultdict
import simplejson as json
import gzip
import pickle
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AmazonNormalizer:
    """
    This class holds a few utility functions for loading and manipulating
    reviews from the Amazon Reviews dataset. It exposes functionality for 
    doing a few things:
    
    1. Obtaining one-hot encodings of word-level tokens in the Amazon dataset.
       The size of the vocabulary can be customized, as well as the number of
       high-frequency tokens to ignore, if any.
    2. Translating a one-hot encoded document back into a list of plain-text
       tokens.
       
    In order to achieve the above, AmazonNormalizer also does the following
    handy things:
    
    * text normalization: lowercase enforcement and censoring of most 
        characters that are not alphanumeric or punctuation. see normalize().
    * tokenization: on whitespace, punctuation separate from alphanumeric strings, with 
        special attention to apostrophe. Does not preserve emoticons, etc.
        See tokenize().
    * token frequency counts: count token frequencies in (a subset of) the 
        corpus. These can be used to set the vocabularies for one-hot encoding.
    * persistence: the vocabularies and frequency counts used can be pickled off and reloaded
        if needed
    """

    def __init__(self, reviews_path = "/data/amazon/reviews_Health_and_Personal_Care.json.gz"):
        """
        Arguments
            reviews_path (string): path to a gzipped json file with SNAP Amazon Reviews data
            
        """
        self.reviews_path = reviews_path
        self.counts = Counter()
        self.text = []
        self.idx_to_word = None
        self.word_to_idx = None
        self.punct_re = re.compile(ur'''([,;.!?:"/\|_@#$%^&*~`+-=<>()\[\]\{\}]|'''
                          ur'''(?<!\w)'(?=\w)|(?<=\w)'(?!\w)|(?<!\w)'(?!\w))''')   # hellish cases involving single quotes
        self.space_re = re.compile(r"\s+")
    
    def normalize(self, txt):
        """
        Text normalization
        
        Arguments
            txt (string)
            
        Returns
            (string) like txt, but all lowercase and any character not in keep_chars
                replaced with a space
        """
        keep_chars = set(ur"""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'"/\|_@#$%^&*~`+-=<>()[]{} """ + "\n")
        return ''.join([ c if c in keep_chars else ' ' for c in txt.lower() ])

    def tokenize(self, txt):
        """
        Tokenization. Uses whitespace, mostly, but each punctuation
        mark also gets its own token, except for single quote
        (apostrophe) when surrounded by alphanumeric characters.
        
        Arguments
            txt (string): to be tokenized
        
        Returns
            (list) each element is a string representing an individual token
                of txt.
        """
        return self.space_re.sub(" ",
                self.punct_re.sub(" \\1 ", txt)).split()

    def review_text(self, json_line):
        """
        Parse a single JSON object rerpesenting an Amazon product review,
        retrieving the review text.
        
        Arguments
            json_line (string): a valid JSON object
            
        Returns
            (string) the text of an Amazon product review
        """
        json_obj = json.loads(json_line)
        review = json_obj['reviewText']
        review_text = review.lower()
        return review_text
    
    def load_reviews(self):
        """
        Amazon-specific.
        Loads all reviews, tokenized, into self.text, a list of tokens.
        This is a prelude to counting these tokens.
        This may be memory-intensive; it is recommended that
        self.reviews_path be pointed to a subset of the reviews dataset.
        """
        i = 0
        with gzip.open(self.reviews_path) as f:
            for line in f:
                tokenized_review = tokenize_amazon(
                                    normalize_amazon(
                                    self.review_text(line)))
                
                self.text.extend(tokenized_review)
                i += 1
                if i % 10000 == 0:
                    logger.info("Record {}...".format(i))
                    
    
    def count_words(self):
        """
        Count tokens in self.text
        
        Side effect
            self.counts is a Counter (dict-like) whose keys
            are the unique tokens found in self.text and 
            whose values are the number of times each token occurs.
        """
        if self.text == []:
            self.load_reviews()
        self.counts = Counter(self.text)
    
    def make_dictionaries(self,vocab_size=5000, exclude_top=25):
        """
        Produce dict-like objects that are necessary to encode
        plain text in vector form.
        """
        counts = self.counts
        cutoff_point = vocab_size + exclude_top
        top_n_words = zip(*counts.most_common(cutoff_point))[0]
        try:
            stop_words = zip(*counts.most_common(exclude_top))[0]
        except IndexError:
            stop_words = []
        keep_words = [ w for w in top_n_words if w not in stop_words ]
        self.word_to_idx = { w: i + 1 for i, w in enumerate(keep_words) }
        self.idx_to_word = defaultdict(str, { i: w for i, w in enumerate(keep_words) })
        self.onehot_lookup = np.concatenate((np.zeros((vocab_size, 1)), np.identity(vocab_size)), axis=1)

    def to_disk(self,dump_path):
        pickle.dump((self.idx_to_word, self.word_to_idx, self.counts), file=file(dump_path,"w"))
        
    def from_disk(self,dump_path):
        self.idx_to_word, self.word_to_idx, self.counts = pickle.load(file=file(dump_path,"r"))
        vocab_size = len(self.idx_to_word)
        self.onehot_lookup = np.concatenate((np.zeros((vocab_size, 1)), np.identity(vocab_size)), axis=1)

    def to_onehot_word(self,txt):
        """
        Converts plain text, tokenized or not, to one-hot (by word)
        encoding. make_dictionaries must be run before this.
        
        Arguments
            txt (list or string): if list, should be list of tokens; if string,
                will be tokenized before proceeding
                
        Returns
            (numpy.ndarray, 2-D): a vocab_size x number of tokens one-hot 
            encoding of the tokens in txt,  according to the parameters 
            given to make_dictionaries. Out-of-vocabulary tokens are represented
            as all-zero vectors.
        """
        # tokenize if necessary (but does not normalize, beware)
        if type(txt) != list:
            txt = self.tokenize(txt)

        keep_words = self.word_to_idx.keys()
        vocab_size = len(keep_words)
        indices = [ self.word_to_idx[w] if w in keep_words else 0 for w in txt ]

        return onehot_lookup[:,indices]

    def to_text(self,onehot_word):
        """
        Converts a one-hot encoding into a list of plain-text
        tokens. make_dictionaries must be run before using this function.
        
        Arguments
            onehot_word (numpy.adarray, 2-D): one-hot encoding of 
            tokens in document, vocab_size x num_tokens
            
        Returns
            (list) plain-text tokens; all-zero (out-of-vocab) tokens represented as 
            empty string
        """
        
        return [self.idx_to_word[i] for i in 
                [ np.asscalar(a[0]) if a[0].size != 0 else None for a in 
                [ np.nonzero(onehot_word[:,x]) for x in range(onehot_word.shape[1]) ] ] ]
    

if __name__=="__main__":
    # Usage demo:
    a = DocumentNormalizer()
    a.load_reviews()
    a.count_words()
    a.make_dictionaries(vocab_size=10000, exclude_top=0)
    # demo serialization
    a.to_disk("/data/amazon/amazon_normalizer.pkl")
    a = None
    a = DocumentNormalizer()
    a.from_disk("/data/amazon/amazon_normalizer.pkl")
    # tokenization, normalization, and one-hot encoding
    test_str ="dkjasdlkajsdlk hahah I am a product of quality."
    test_tokens = a.tokenize(a.normalize(test_str))
    logger.info(test_tokens)
    logger.info(a.to_onehot_word(test_tokens).sum(axis=0))
    logger.info([ a.word_to_idx.get(x, None) for x in test_tokens])

