import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
import pandas as pd
from sklearn.model_selection import train_test_split
import dill
#from configparser import ConfigParser

from config import get_args

args = get_args()

#parser = ConfigParser()
#parser.read('config.ini')

en = spacy.load('en')
de = spacy.load('de')

def data_load():

    data_en = open('data/en-de/train.en', encoding='utf-8').read().split('\n')
    data_de = open('data/en-de/train.de', encoding='utf-8').read().split('\n')

    raw_data = {'EN' : [line for line in data_en], 'DE': [line for line in data_de]} 

    return raw_data

def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]
def tokenize_de(sentence):
    return [tok.text for tok in de.tokenizer(sentence)]

def data_tokenize_save():

    print('raw data loading...')

    raw_data = data_load()

    df = pd.DataFrame(raw_data, columns=["EN", "DE"])

    df['en_len'] = df['EN'].str.count(' ')
    df['de_len'] = df['DE'].str.count(' ')
    df = df.query('de_len < 80 & en_len < 80') 
    df = df.query('de_len < en_len * 1.5 & de_len * 1.5 > en_len')

    train, val = train_test_split(df, test_size=0.2)
    train.to_csv("data/en-de/train.csv", index=False)
    val.to_csv("data/en-de/val.csv", index=False)

    print('tokenizing...') 

    data_fields = [('EN', EN_TEXT), ('DE', DE_TEXT)]
    train,val = TabularDataset.splits(path='data/en-de', train='train.csv', validation='val.csv', format='csv', fields=data_fields)

    with open('data/en-de/train.Field', 'wb') as f:
        dill.dump(train, f)
    with open('data/en-de/val.Field', 'wb') as f:
        dill.dump(val, f)

    return train,val

EN_TEXT = Field(tokenize=tokenize_en, init_token = "<sos>", eos_token = "<eos>")
DE_TEXT = Field(tokenize=tokenize_de, init_token = "<sos>", eos_token = "<eos>")


#if(parser.getboolean('data','data_load')):


if(args.preprocess):

    train,val = data_tokenize_save()

else:
    
    print('data loading...')

    with open('data/en-de/train.Field', 'rb') as f:
        train = dill.load(f) 
    with open('data/en-de/val.Field', 'rb') as f:
        val = dill.load(f) 

print('build vocab...')

EN_TEXT.build_vocab(train, val)
DE_TEXT.build_vocab(train, val)
# len(EN_TEXT.vocab)
batchSize = args.batch_size

train_iter = BucketIterator(train, batch_size=batchSize,sort_key=lambda x: len(x.EN), shuffle=True)
val_iter = BucketIterator(val, batch_size=batchSize,sort_key=lambda x: len(x.EN), shuffle=True)