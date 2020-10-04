python preprocess.py -include_flt
CUDA_VISIBLE_DEVICES=1 python train.py -learning_rate 0.001 -no_share_emb_layout_encoder -seprate_encoder -batch_size 64 -max_grad_norm 0.1 -layout_weight 1 -optim adam 

CUDA_VISIBLE_DEVICES=1 python train.py -learning_rate 0.001 -no_share_emb_layout_encoder -seprate_encoder -batch_size 64 -layout_weight 1 -optim adam -epochs 6

-dropout 0.3 -dropout_i 0.3
CUDA_VISIBLE_DEVICES=1 python evaluate.py -model_path=swbdIO/run.9/m_* -output_file run.9/test

CUDA_VISIBLE_DEVICES=7 python evaluate.py -model_path=swbdIO/run.2/m_* -output_file run.2/test_gold -gold_layout

CUDA_VISIBLE_DEVICES=7 python train.py -encoder_type transformer -transformer_learning_rate 0.0001 -batch_size 64 -layout_weight 1 -optim adam -report_every 1000 -gelu