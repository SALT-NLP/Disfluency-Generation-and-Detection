from __future__ import division
import os
import argparse
import torch
import codecs
import glob
from nltk.translate.bleu_score import corpus_bleu

import table
import table.IO
import opts

parser = argparse.ArgumentParser(description='evaluate.py')
opts.translate_opts(parser)
opt = parser.parse_args()
#torch.cuda.set_device(opt.gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

opt.dataset = opt.dataset + opt.tag_type
opt.anno = os.path.join(opt.root_dir, opt.dataset, '{}.txt'.format(opt.split))

if opt.beam_size > 0:
    opt.batch_size = 1

print('trans_opt:',vars(opt))

def main():
    dummy_parser = argparse.ArgumentParser(description='train.py')
    opts.model_opts(dummy_parser)
    opts.train_opts(dummy_parser)
    dummy_opt = dummy_parser.parse_known_args([])[0]

    js_list = table.IO.read_anno(opt.anno, opt)
    print('data_len',len(js_list))

    metric_name_list = ['lay','tgt']
    with open(os.path.join(opt.root_dir, opt.dataset, opt.output_file), 'a', encoding='utf-8') as writer:
        #writer.write('{}\n'.format(vars(translator.model_opt)))
        writer.write('trans_opt: {}\n'.format(vars(opt)))
    prev_best = (None, None, None, None)
    for fn_model in glob.glob(opt.model_path):
        opt.model = fn_model
        print(fn_model)
        #print(opt.anno)

        translator = table.Translator(opt, dummy_opt.__dict__)
        data = table.IO.TableDataset(
            js_list, translator.fields, translator.model_opt, test=True)


        test_data = table.IO.OrderedIterator(
            dataset=data, device=device, batch_size=opt.batch_size, train=False, sort=True, sort_within_batch=False)

        # inference
        r_list = []
        with torch.no_grad():
            for batch in test_data:
                r = translator.translate(batch,js_list)
                r_list += r

        r_list.sort(key=lambda x: x.idx)
        assert len(r_list) == len(js_list), 'len(r_list) != len(js_list): {} != {}'.format(
            len(r_list), len(js_list))

        # evaluation
        for pred, gold in zip(r_list, js_list):
            pred.eval(gold, gold_diversity=opt.gold_diversity if 'gold_diversity' in opt.__dict__ else False)
        with open(os.path.join(opt.root_dir, opt.dataset, opt.output_file), 'a', encoding='utf-8') as writer:
            writer.write('{}\n'.format(fn_model))
            print('Results:\n')
            for metric_name in metric_name_list:
                c_correct = sum((x.correct[metric_name] for x in r_list))
                acc = c_correct / len(r_list)
                print('{}: {} / {} = {:.2%}\n'.format(metric_name,
                                                c_correct, len(r_list), acc))
                writer.write('{}: {} / {} = {:.2%}\n'.format(metric_name,
                                                c_correct, len(r_list), acc))
                if metric_name == 'tgt' and (prev_best[0] is None or acc > prev_best[1]):
                    prev_best = (fn_model, acc, r_list, translator.model_opt)
            print('disf_less_than_one:', sum((x.disflen_lessthanone for x in r_list)))
            onegrams=[]
            twograms=[]
            for x in r_list:
                onegrams.extend(x.one_grams)
                twograms.extend(x.two_grams)
            c_correct = sum(onegrams)
            acc = c_correct / len(onegrams)
            print('{}: {} / {} = {:.2%}'.format('onegram',
                                            c_correct, len(onegrams), acc))
            writer.write('{}: {} / {} = {:.2%}\n'.format('onegram',
                                            c_correct, len(onegrams), acc))
            c_correct = sum(twograms)
            acc = c_correct / len(twograms)
            print('{}: {} / {} = {:.2%}'.format('twogram',
                                            c_correct, len(twograms), acc))
            writer.write('{}: {} / {} = {:.2%}\n'.format('twogram',
                                            c_correct, len(twograms), acc))
            references=[]
            candidates=[]
            for x in r_list:
                references.append([x.gold_tgt])
                candidates.append(x.tgt)
            blue_score = corpus_bleu(references, candidates)
            print('BLUE:',blue_score)
            writer.write('{} = {:.5}\n'.format('BLUE', blue_score))

    if (prev_best[0] is not None):
        print("Writing to "+os.path.join(opt.root_dir, opt.dataset, opt.output_file))
        disf_generated=0
        with open(os.path.join(opt.root_dir, opt.dataset, opt.output_file+'_generated'), 'w', encoding='utf-8') as writer:
            for x in prev_best[2]:
                if 'I' in x.tgt_tags:
                    disf_generated+=1
                    assert (len(x.tgt) == len(x.tgt_tags))
                    writer.write('\t'.join(x.tgt) + '\n')
                    writer.write('\t'.join(['P'] * len(x.tgt_tags)) + '\n')
                    writer.write('\t'.join(x.tgt_tags) + '\n')
                    writer.write('\n')
        with open(os.path.join(opt.root_dir, opt.dataset, opt.output_file+'_out'), 'w', encoding='utf-8') as writer:
            for x in prev_best[2]:
                writer.write(' '.join(x.tgt) + '\n')
        print('Disf_sents/Total_sents',disf_generated,len(prev_best[2]))

if __name__ == "__main__":
    main()
