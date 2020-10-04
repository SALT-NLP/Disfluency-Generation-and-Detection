
TASK=swbdIO

: '
for SPLIT in train dev
do
  for LANG in flt disf
  do
    python -m examples.roberta.multiprocessing_bpe_encoder \
    --encoder-json /home/jyang690/style/encoder.json \
    --vocab-bpe /home/jyang690/style/vocab.bpe \
    --inputs "$TASK/$LANG.$SPLIT" \
    --outputs "$TASK/$SPLIT.bpe.$LANG" \
    --workers 60;
  done
done

fairseq-preprocess \
  --source-lang "flt" \
  --target-lang "disf" \
  --trainpref "${TASK}/train.bpe" \
  --validpref "${TASK}/dev.bpe" \
  --destdir "${TASK}/bin/" \
  --workers 60 \
  --srcdict /home/jyang690/style/dict.txt \
  --tgtdict /home/jyang690/style/dict.txt; 



#TOTAL_NUM_UPDATES=3000
TOTAL_NUM_EPOCHS=30
#20000  
WARMUP_UPDATES=500
#500     
LR=1e-5
#3e-05
MAX_TOKENS=2048
#2048
UPDATE_FREQ=4
#32
BART_PATH=/nethome/jyang690/style/bart.large/model.pt

CUDA_VISIBLE_DEVICES=$1 python /home/jyang690/style/fairseq/train.py "${TASK}/bin" \
    --restore-file $BART_PATH \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang flt --target-lang disf \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --max-epoch $TOTAL_NUM_EPOCHS --warmup-updates $WARMUP_UPDATES\
    --memory-efficient-fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters \
    --save-dir "$TASK/bart-checkpoints";'


for f in $TASK/bart-checkpoints/checkpoint*
do 
    echo '------------------------------------------------------------'
    echo $f
    CUDA_VISIBLE_DEVICES=$1 python /home/jyang690/style/bart_gen.py -checkpoint $(dirname $f) -model $(basename $f) -src $TASK/flt.test -tgt $TASK/disf.test.pred_bart -data "${TASK}/bin"
    python calculate_bleu.py $TASK/disf.test.pred_bart $TASK/disf.test 
    python calculate_acc.py $TASK/disf.test.pred_bart $TASK/disf.test
done
