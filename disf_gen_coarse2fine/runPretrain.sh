for f in swbdIO/run.18/m_*
do 
    echo '------------------------------------------------------------' 2>&1 | tee -a evalTransformerPretrain.txt
    echo $f 2>&1 | tee -a evalTransformerPretrain.txt
    python preprocess.py -no_disf_trans -disf_seg -include_flt
    CUDA_VISIBLE_DEVICES=7 python train.py -encoder_type transformer -report_every 1000 -transformer_learning_rate 0.00005 -gelu -train_from $f -train_from_pretrain -start_checkpoint_at 50 2>&1 | tee -a evalTransformerPretrain.txt
done
