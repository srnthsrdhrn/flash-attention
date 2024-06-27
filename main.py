from argparse import ArgumentParser
from bert_pytorch.model.bert import BERT
from huggingface_hub import hf_hub_download

dataset = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
)

def main(args):
    BERT()

if __name__  == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--attn_type',choices=['normal','flash'],default='normal')
    args = arg_parser.parse_args()
    main(args)