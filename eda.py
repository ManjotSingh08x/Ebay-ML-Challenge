import csv 
import pandas as pd
import numpy as np
import pytorch
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("GottBERT/GottBERT_base_last")
model = AutoModelForMaskedLM.from_pretrained("GottBERT/GottBERT_base_last")
print(model)
TRAIN_PATH = 'data/ebay/Tagged_Titles_Train.tsv'
LIST_PATH = 'data/ebay/Listing_Titles.tsv'
tagged_train = pd.read_csv(TRAIN_PATH, keep_default_na=False, na_values=None, sep='\t')
listing_titles =pd.read_csv(LIST_PATH, keep_default_na=False, na_values=None, sep='\t')
print(tagged_train.head(10))
print()
print(listing_titles.head(10))