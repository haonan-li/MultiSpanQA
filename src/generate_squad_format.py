# To prepare single span training data
# v1: single answer span is the shortest long span that includes all answer spans.
# v2: single answer span is one of the answer spans, create n examples for multi-span example with n answers.

import os
import json
from copy import deepcopy
from tqdm import tqdm_notebook
import uuid

def prepare_softmax_training_v1(data_dir='../data/MultiSpanQA_data', data_file='train.json'):
    with open(os.path.join(data_dir, data_file)) as f:
        data = json.load(f)

    for piece in data['data']:
        label = piece['label']
        st = label.index('B')
        for i,l in enumerate(label[::-1]):
            if l!='O':
                break
        ed = len(label) - i
        label[st] = 'B'
        for i in range(st+1,ed):
            label[i] = 'I'
        piece['label'] = label

    with open(os.path.join(data_dir,'train_softmax_v1.json'),'w') as f:
        json.dump(data, f)


def prepare_softmax_training_v2(data_dir='../data/MultiSpanQA_data', data_file='train.json'):
    def my_index(l,target,start=0):
        if start == -1:
            return -1
        for i in range(start, len(l)):
            if l[i] == target:
                return i
        return -1

    with open(os.path.join(data_dir, data_file)) as f:
        data = json.load(f)
    new_data = []
    id = 0
    for piece in data['data']:
        st = 0
        ed = 0
        label = piece['label']
        while st != -1:
            st = my_index(label,'B', ed)
            ed = my_index(label,'O', st)
            if st != -1:
                new_label = ['O'] * len(label)
                new_label[st] = 'B'
                for i in range(st+1,ed):
                    new_label[i] = 'I'
                new_piece = deepcopy(piece)
                new_piece['label'] = new_label
                new_piece['id'] = piece['id']+str(id)
                new_data.append(new_piece)
                id += 1
    data['data'] = new_data
    with open(os.path.join(data_dir,'train_softmax_v2.json'),'w') as f:
        json.dump(data, f)


def create_squad_format(file_path):
    """ Create squad format for using huggingface run_squad.py script """
    fpath,fname = os.path.split(file_path)

    with open(file_path) as f:
        data = json.load(f)
    new_data = []
    for piece in data['data']:
        id = piece['id']
        context = piece['context']
        if "label"in piece:
            label = piece['label']
            st = label.index('B', 0) if 'B' in label else -1
        if ('test' in fname) or st == -1:
            new_data.append({'question':' '.join(piece['question']), 'context':' '.join(context),'answers':{"text": [], "answer_start": []}, 'id':id})
        else:
            para = {
                "paragraphs": [
                    {
                        "qas": [{
                            "question": ' '.join(piece['question']),
                            "id": id,
                            "answers": [],
                            "is_impossible": False
                        }],
                        "context": ' '.join(context),
                        "document_id": id}]}

            if st != -1:
                text = ""
                for idx, tag in enumerate(label):

                    if tag == "B":
                        text = context[idx]
                        char_st = len(' '.join(context[:idx]))
                    if tag == "I":
                        text += " "+ context[idx]

                    if len(text) > 0 and tag == "O":
                        answer_id = uuid.uuid4().hex

                        answer = {"text": text, "answer_start": char_st + 1, "answer_id": answer_id, "document_id": id}
                        para["paragraphs"][0]["qas"][0]["answers"].append(answer)
                        text = ""

            new_data.append(para)



    data['data'] = new_data
    with open(os.path.join(fpath, 'squad_'+fname),'w') as f:
        json.dump(data, f, indent=4)

def prepare_softmax_training_expand(multi_dir='../data/MultiSpanQA_data', expand_dir='../data/MultiSpanQA_expand_data', data_file='train.json'):
    """ Merge single span version of MultiSpanQA with expanded no/single answer examples """
    with open(os.path.join(expand_dir,data_file)) as f:
        ori_expand = json.load(f)
    with open(os.path.join(expand_dir,data_file)) as f:
        ori_multi = json.load(f)
        ori_multi_ids = set(list(map(lambda x: x['id'], ori_multi['data'])))
    no_multi_data = list(filter(lambda x: not x['id'] in ori_multi_ids, ori_expand['data']))
    with open(os.path.join(expand_dir,'train_softmax_v1.json'), 'w') as f:
        with open(os.path.join(multi_dir,'train_softmax_v1.json')) as f1:
            ori_expand['data'] = no_multi_data + json.load(f1)['data']
            json.dump(ori_expand, f)
    with open(os.path.join(expand_dir,'train_softmax_v2.json'), 'w') as f:
        with open(os.path.join(multi_dir,'train_softmax_v2.json')) as f1:
            ori_expand['data'] = no_multi_data + json.load(f1)['data']
            json.dump(ori_expand, f)

def main():
    # use default dir and files
    prepare_softmax_training_v1()
    prepare_softmax_training_v2()
    prepare_softmax_training_expand()

    file_list = [
                 '../data/MultiSpanQA_data/train_softmax_v1.json',
                 '../data/MultiSpanQA_data/train_softmax_v2.json',
                 '../data/MultiSpanQA_data/valid.json',
                 '../data/MultiSpanQA_data/test.json',
                 '../data/MultiSpanQA_expand_data/train_softmax_v1.json',
                 '../data/MultiSpanQA_expand_data/train_softmax_v2.json',
                 '../data/MultiSpanQA_expand_data/valid.json',
                 '../data/MultiSpanQA_expand_data/test.json'
                 ]
    for f in file_list:
        create_squad_format(f)

main()
