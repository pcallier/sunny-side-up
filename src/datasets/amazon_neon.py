# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Defines basic input datatset types.
"""
import numpy as np
import h5py
import simplejson as json
import gzip
from neon import NervanaObject

# parameters: data modeling
# path_to_amazon = '/mnt/data/AmazonReviews/aggressive_dedup.json.gz'
# total_records = 82836502
# path_to_amazon = '/mnt/data/AmazonReviews/reviews_Health_and_Personal_Care.json.gz'
# total_records =  2982356


def text_to_uint8(text_string):
    '''
    Converts a text string into a byte array, but only ascii characters 32-64, 91-126
    maps them down to 0-68, and everything else to 0 (non-printables become space)
    '''
    # Can be much simpler if including caps:  a -= 31; a[a>95] = 0
    a = np.array(bytearray(text_string)) - 32
    a[a > 32] = -a[a > 32] - 127
    a[a > 68] = 0
    return a


def uint8_to_text(uary):
    '''
    Reverses mapping of text_to_uint8
    '''
    # Can be much simpler if including caps:  uary += 32
    uary[uary > 32] = -uary[uary > 32] - 95
    uary[uary < 34] = uary[uary < 34] + 32
    return str(bytearray(uary))


class AmazonBatchWriter(object):
    def __init__(self, data_path, out_file, min_length=100, max_length=1014):
        self.data_path = data_path
        self.min_length = min_length
        self.max_length = max_length
        self.record_dim = max_length + 1
        self.ndata = 0
        self.ofile = h5py.File(out_file, "w")
        self.output = self.ofile.create_dataset(name="reviews",
                                                shape=(2**20, self.record_dim),
                                                maxshape=(None, self.record_dim),
                                                dtype=np.uint8,
                                                chunks=(2048, self.record_dim))
        self.cache_size = 2048*16
        self.rbuf = np.empty((self.cache_size, self.max_length), dtype=np.uint8)
        self.lbuf = np.empty((self.cache_size,), dtype=np.uint8)

    def __del__(self):
        # Write out the remainder
        idx = self.ndata % self.cache_size
        self.dump_cache(self.output, self.rbuf, self.lbuf, idx)
        self.output.resize(self.ndata, axis=0)
        self.ofile.close()

    def parse_review(self, json_line):
        json_obj = json.loads(json_line)
        score = float(json_obj['overall'])
        review = json_obj['reviewText']
        if score == 3.0 or len(review) < self.min_length:
            return None
        overall = 0 if score < 3.0 else 1
        review_text = review.lower()[:self.max_length].ljust(self.max_length)
        return review_text, overall

    def run(self):
        with gzip.open(self.data_path, "r") as f:
            for line in f:
                parsed_record = self.parse_review(line)
                if parsed_record is None:  # Uninformative Review
                    continue
                else:
                    idx = self.ndata % self.cache_size
                    self.rbuf[idx] = bytearray(parsed_record[0])
                    self.lbuf[idx] = parsed_record[1]
                    self.ndata += 1
                    if (self.ndata % self.cache_size) == 0:
                        self.dump_cache(self.output, self.rbuf, self.lbuf, self.cache_size)

    def dump_cache(self, hd5set, reviewbuf, labelbuf, offset):
        reviewbuf -= 32
        reviewbuf[reviewbuf > 32] = -reviewbuf[reviewbuf > 32] - 127
        reviewbuf[reviewbuf > 68] = 0
        hd5set[(self.ndata - offset):self.ndata, 0] = labelbuf[:offset]
        hd5set[(self.ndata - offset):self.ndata, 1:] = reviewbuf[:offset]
        print "{} reviews processed".format(self.ndata)


class AmazonDataIterator(NervanaObject):

    def __init__(self, fname, nvocab=69, nlabels=2):
        """
        Implements loading of given data into backend tensor objects. If the
        backend is specific to an accelarator device, the data is copied over
        to that device.

        Args:
            fname (string): hdf5 file containing reviews and sentiment labels
            nlabels (int, optional): The number of possible types of labels. 2 for binary good/bad
            nvocab  (int, optional): Tne number of letter tocans
                (not necessary if not providing labels)
            nlabels (Int)

        """
        # Treat singletons like list so that iteration follows same syntax
        self.ofile = h5py.File(fname, "r")
        self.dbuf = self.ofile['reviews']
        self.ndata = self.dbuf.shape[0]
        self.start = 0

        self.nlabels = nlabels
        self.nvocab = nvocab
        self.nsteps = self.dbuf.shape[1] - 1  # removing 1 for the label at the front

        # on device tensor for review chars and one hot
        self.xlabels_flat = self.be.iobuf((1, self.nsteps), dtype=np.int32)
        self.xlabels = self.xlabels_flat.reshape((self.nsteps, self.be.bsz))
        self.Xbuf_flat = self.be.iobuf((self.nvocab, self.nsteps))
        self.Xbuf = self.Xbuf_flat.reshape((self.nvocab * self.nsteps, self.be.bsz))

        self.ylabels = self.be.iobuf(1, dtype=np.int32)
        self.ybuf = self.be.iobuf(self.nlabels)

        # This makes upstream layers interpret each example as a 1d image
        self.shape = (1, self.nvocab, self.nsteps)
        self.Xbuf.lshape = self.shape  # for backward compatibility

    @property
    def nbatches(self):
        return -((self.start - self.ndata) // self.be.bsz)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        Relevant for when one wants to call repeated evaluations on the dataset
        but don't want to wrap around for the last uneven minibatch
        Not necessary when ndata is divisible by batch size
        """
        self.start = 0

    def __iter__(self):
        """
        Defines a generator that can be used to iterate over this dataset.

        Yields:
            tuple: The next minibatch. A minibatch includes both features and
            labels.
        """
        for i1 in range(self.start, self.ndata, self.be.bsz):
            i2 = min(i1 + self.be.bsz, self.ndata)
            bsz = i2 - i1
            if i2 == self.ndata:
                self.start = self.be.bsz - bsz

            self.xlabels[:] = self.dbuf[i1:i2, 1:].T.copy()
            self.ylabels[:] = self.dbuf[i1:i2, 0:1].T.copy()
            if self.be.bsz > bsz:
                self.xlabels[:, bsz:] = self.dbuf[:self.start, 1:].T.copy()
                self.ylabels[:, bsz:] = self.dbuf[:self.start, 0:1].T.copy()

            self.Xbuf_flat[:] = self.be.onehot(self.xlabels_flat, axis=0)
            self.ybuf[:] = self.be.onehot(self.ylabels, axis=0)
            yield (self.Xbuf, self.ybuf)


if __name__ == '__main__':
    h5file = '/root/data/pcallier/amazon/temp.hd5'
    amzn_path = '/root/data/pcallier/amazon/reviews_Health_and_Personal_Care.json.gz'
    azbw = AmazonBatchWriter(amzn_path, h5file)
    azbw.run()

    from neon.backends.nervanagpu import NervanaGPU
    ng = NervanaGPU(0, device_id=1)

    NervanaObject.be = ng
    ng.bsz = 128
    train_set = AmazonDataIterator(h5file)
    for bidx, (X_batch, y_batch) in enumerate(train_set):
        reviewnum = input("Pick review index to fetch and decode")
        binreview = X_batch.get().T[reviewnum].reshape(69, -1).argmax(axis=0).astype(np.uint8)
        print uint8_to_text(binreview), y_batch.get().T[reviewnum].argmax(axis=0)
