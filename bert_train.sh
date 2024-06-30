#!/bin/bash
# bert-vocab -c data/corpus.small -o data/vocab.small
source ~/.bashrc
DEFAULT_EPOCHS=1
DEFAULT_BATCH_SIZE=32
DEFAULT_HIDDEN_DIMENSION=256
DEFAULT_LAYERS=8
DEFAULT_ATTN_HEADS=8
CORPUS=corpus.mini.txt
VOCAB=vocab.mini

echo "Varying Batch Size"
for attn_type in flash_attn regular_attention;
do
    echo "Attention Type $attn_type"
    for batch_size in 256;
    do
        echo "Batch Size: $batch_size"
        poetry run bert -c data/$CORPUS -v data/$VOCAB -o output/bert_small.model --batch_size $batch_size --epochs $DEFAULT_EPOCHS --hidden $DEFAULT_HIDDEN_DIMENSION --layers $DEFAULT_LAYERS --attn_heads $DEFAULT_ATTN_HEADS --attn_type $attn_type
    done
done

echo "Varying number of  layers"
for attn_type in flash_attention regular_attention;
do
    echo "Attention Type $attn_type"
    for layers in 4 8 16 32 64 128 256 512;
    do
        echo "Layer: $layers"
        bert -c data/$CORPUS -v data/$VOCAB -o output/bert_small.model --batch_size $DEFAULT_BATCH_SIZE --epochs $DEFAULT_EPOCHS --hidden $DEFAULT_HIDDEN_DIMENSION --layers $layers --attn_heads $DEFAULT_ATTN_HEADS --attn_type $attn_type
    done
done

echo "Varying hidden dimensions"
for attn_type in flash_attention regular_attention;
do
    echo "Attention Type $attn_type"
    for hidden_dimensions in 64 128 256 512;
    do
        echo "Hidden Dimension: $hidden_dimensions"
        bert -c data/$CORPUS -v data/$VOCAB -o output/bert_small.model --batch_size $DEFAULT_BATCH_SIZE --epochs $DEFAULT_EPOCHS --hidden $hidden_dimensions --layers $DEFAULT_LAYERS --attn_heads $DEFAULT_ATTN_HEADS --attn_type $attn_type
    done
done

echo "Varying Attention Heads"
for attn_type in flash_attention regular_attention;
do
    echo "Attention Type $attn_type"
    for attn_heads in 8 16 32 64 128 256 512;
    do
        echo "Attention Heads: $attn_heads"
        bert -c data/$CORPUS -v data/$VOCAB -o output/bert_small.model --batch_size $DEFAULT_BATCH_SIZE --epochs $DEFAULT_EPOCHS --hidden $DEFAULT_HIDDEN_DIMENSION --layers $DEFAULT_LAYERS --attn_heads $attn_heads --attn_type $attn_type
    done
done

echo "Varying epochs"
for attn_type in flash_attention regular_attention;
do
    echo "Attention Type $attn_type"
    for epochs in 5 10 15 20;
    do
        echo "Epochs: $epochs"
        bert -c data/$CORPUS -v data/$VOCAB -o output/bert_small.model --batch_size $DEFAULT_BATCH_SIZE --epochs $epochs --hidden $DEFAULT_HIDDEN_DIMENSION --layers $DEFAULT_LAYERS --attn_heads $DEFAULT_ATTN_HEADS --attn_type $attn_type
    done
done

