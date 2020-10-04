# Disfluency-Generation-and-Detection
This repo contains codes for the following paper: 

*Jingfeng Yang, Zhaoran Ma, Diyi Yang*: MixText: Linguistically-Informed Interpolation of Hidden Space for Semi-Supervised Text Classification. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP'2020)

If you would like to refer to it, please cite the paper mentioned above. 


## Getting Started
These instructions will get you running the codes.

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* Pytorch_transformers (also known as transformers)

### Planner and Generator Disfluency Generation

```
cd disf_gen_coarse2fine &&
python train.py -learning_rate 0.001 -no_share_emb_layout_encoder -seprate_encoder -batch_size 64 -max_grad_norm 0.1 -layout_weight 1 -optim adam &&
python evaluate.py &&
cd ..
```

### Heuristic Planner + GPT2 Generator for data augmentation

```
CUDA_VISIBLE_DEVICES=0 python transformers/examples/run_language_modeling.py --output_dir=news3m_ml_finetune_st --model_type=gpt2 --model_name_or_path=gpt2 --do_train --train_data_file=news_3m --do_eval --eval_data_file=swbd_LM_val --line_by_line --eval_all_checkpoints --num_train_epochs 6 --logging_steps 6000 --save_steps 6000 &&
python createFakeLMdist.py -infile news_to_fake_3m -outfile news_fake_3m_newstune360000_mp -model_path news3m_ml_finetune_st/checkpoint-360000 -gpu 2222333333555555 &&
python writePretrain.py
```


### Disfluency detection w/ or w/o augmented data
Please run `./code/train.py` to train the MixText model (use both labeled and unlabeled training data):
```
python trainBertPretrain.py ||
python trainBertPretrain.py -p
```

## Aknowledgement

Disfluency generation code is adapted from [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) and [Coarse2fine Semantic Parsing](https://github.com/donglixp/coarse2fine)


