# For data loading.
from torchtext import data, datasets
from torchtext.data import TabularDataset

if True:
    import spacy
    spacy_en = spacy.load('en')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"    
    SOS_TOKEN = "<s>"
    EOS_TOKEN = "</s>"
    LOWER = True
    
    # we include lengths to provide to the RNNs
    SRC = data.Field(tokenize=tokenize_en, 
                     batch_first=True, lower=LOWER, include_lengths=True,
                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None, eos_token=EOS_TOKEN)
    TRG = data.Field(tokenize=tokenize_en, 
                     batch_first=True, lower=LOWER, include_lengths=True,
                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN, eos_token=EOS_TOKEN)

    #MAX_LEN = 25  # NOTE: we filter out a lot of sentences for speed
    tv_datafields = [("src", SRC), ("trg", TRG)]
    train_data, valid_data, test_data = TabularDataset.splits(
                  path="./", # the root directory where the data lies
                  train='CNN_train.csv', validation="CNN_val.csv", test = "CNN_test.csv",
                  format='csv',
                  skip_header=True, # if your csv header has a header, make sure to pass this to ensure it doesn't get proceesed as data!
                  fields=tv_datafields)

    MIN_FREQ = 5  # NOTE: we limit the vocabulary to frequent words for speed
    SRC.build_vocab(train_data.src, min_freq=MIN_FREQ)
    TRG.build_vocab(train_data.trg, min_freq=MIN_FREQ)
    
    PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]


#create training, validation, and test objects
train_iter = data.BucketIterator(train_data, batch_size=16, train=True, 
                                 sort_within_batch=True, 
                                 sort_key=lambda x: (len(x.src), len(x.trg)), repeat=False,
                                 device=DEVICE)
valid_iter = data.Iterator(valid_data, batch_size=1, train=False, sort=False, repeat=False, 
                           device=DEVICE)
test_iter = data.BucketIterator(test_data, batch_size=1, train=False, sort=False, repeat=False,
                                device=DEVICE)


def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.src, batch.trg, pad_idx)