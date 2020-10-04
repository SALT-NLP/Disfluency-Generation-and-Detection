TASK=swbdIO
#onmt_preprocess -train_src $TASK/flt.train -train_tgt $TASK/disf.train -valid_src $TASK/flt.dev  -valid_tgt $TASK/disf.dev -save_data $TASK/train_val -overwrite
#CUDA_VISIBLE_DEVICES=$1 onmt_train -data $TASK/train_val -save_model model/6-fluent-disf/model_sgd -gpu_ranks 0 -valid_steps 500 -train_steps 15000 -save_checkpoint_steps 500
for f in model/6-fluent-disf/model_sgd*
do 
    echo '------------------------------------------------------------'
    echo $f
    CUDA_VISIBLE_DEVICES=$1 onmt_translate -gpu 0 -model $f -src $TASK/flt.test -tgt $TASK/disf.test -replace_unk -output $TASK/disf.test.pred
    python calculate_bleu.py $TASK/disf.test.pred $TASK/disf.test
    python calculate_acc.py $TASK/disf.test.pred $TASK/disf.test
done
