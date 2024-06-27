# %%
import torch
from tqdm import tqdm
from datasets import load_dataset
from bert_pytorch.model.bert import BERT
# %%
# Code to create corpus file
# with open("corpus.txt",'w') as f:
#     prev =None
#     for sent in tqdm(dataset['train']):
#         if prev is not None:
#             f.write(f"{prev}\t{sent['text']}\n")
#         prev = sent['text']

# %%
model = torch.load("output/bert.model.ep9")
# %%
dataset = load_dataset("bookcorpus/bookcorpus",trust_remote_code=True)
# %%
model(dataset['train'][0])
# %%
