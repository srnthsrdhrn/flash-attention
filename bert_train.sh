#!/bin/bash
bert-vocab -c data/corpus.small -o data/vocab.small
bert -c data/corpus.small -v data/vocab.small -o output/bert_small.model --batch_size 160