# Script for MultiSpanQA evaluation
import os
import re
import json
import string
import difflib
import warnings
import numpy as np


def get_entities(label, token):
    def _validate_chunk(chunk):
        if chunk in ['O', 'B', 'I']:
            return
        else:
            warnings.warn('{} seems not to be IOB tag.'.format(chunk))
    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []

    # check no ent
    if isinstance(label[0], list):
        for i,s in enumerate(label):
            if len(set(s)) == 1:
                chunks.append(('O', -i, -i))
    # for nested list
    if any(isinstance(s, list) for s in label):
        label = [item for sublist in label for item in sublist + ['O']]
    if any(isinstance(s, list) for s in token):
        token = [item for sublist in token for item in sublist + ['O']]

    for i, chunk in enumerate(label + ['O']):
        _validate_chunk(chunk)
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):
            chunks.append((' '.join(token[begin_offset:i]), begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
    return chunks


def end_of_chunk(prev_tag, tag):
    chunk_end = False
    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True
    return chunk_end

def start_of_chunk(prev_tag, tag):
    chunk_start = False
    if tag == 'B':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True
    return chunk_start


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_scores(golds, preds, eval_type='em',average='micro'):

    nb_gold = 0
    nb_pred = 0
    nb_correct = 0
    nb_correct_p = 0
    nb_correct_r = 0
    for k in list(golds.keys()):
        gold = golds[k]
        pred = preds[k]
        nb_gold += max(len(gold), 1)
        nb_pred += max(len(pred), 1)
        if eval_type=='em':
            if len(gold) == 0 and len(pred) == 0:
                nb_correct += 1
            else:
                nb_correct += len(gold.intersection(pred))
        else:
            p_score, r_score = count_overlap(gold, pred)
            nb_correct_p += p_score
            nb_correct_r += r_score

    if eval_type == 'em':
        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_gold if nb_gold > 0 else 0
    else:
        p = nb_correct_p / nb_pred if nb_pred > 0 else 0
        r = nb_correct_r / nb_gold if nb_gold > 0 else 0

    f = 2 * p * r / (p + r) if p + r > 0 else 0

    return p,r,f


def count_overlap(gold, pred):
    if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
        return 1,1
    elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
        return 0,0
    p_scores = np.zeros((len(gold),len(pred)))
    r_scores = np.zeros((len(gold),len(pred)))
    for i,s1 in enumerate(gold):
        for j,s2 in enumerate(pred):
            s = difflib.SequenceMatcher(None, s1, s2)
            _,_,longest = s.find_longest_match(0, len(s1), 0, len(s2))
            p_scores[i][j] = longest/len(s2) if longest>0 else 0
            r_scores[i][j] = longest/len(s1) if longest>0 else 0

    p_score = sum(np.max(p_scores,axis=0))
    r_score = sum(np.max(r_scores,axis=1))

    return p_score, r_score


def read_gold(gold_file):
    with open(gold_file) as f:
        data = json.load(f)['data']
        golds = {}
        for piece in data:
            golds[piece['id']] = set(map(lambda x: x[0], get_entities(piece['label'],piece['context'])))
    return golds


def read_pred(pred_file):
    with open(pred_file) as f:
        preds = json.load(f)
    return preds


def multi_span_evaluate_from_file(pred_file, gold_file):
    preds = read_pred(pred_file)
    golds = read_gold(gold_file)
    result = multi_span_evaluate(preds, golds)
    return result


def multi_span_evaluate(preds, golds):
    assert len(preds) == len(golds)
    assert preds.keys() == golds.keys()
    # Normalize the answer
    for k,v in golds.items():
        golds[k] = set(map(lambda x: normalize_answer(x), v))
    for k,v in preds.items():
        preds[k] = set(map(lambda x: normalize_answer(x), v))
    # Evaluate
    em_p,em_r,em_f = compute_scores(golds, preds, eval_type='em')
    overlap_p,overlap_r,overlap_f = compute_scores(golds, preds, eval_type='overlap')
    result = {'em_precision': 100*em_p,
              'em_recall': 100*em_r,
              'em_f1': 100*em_f,
              'overlap_precision': 100*overlap_p,
              'overlap_recall': 100*overlap_r,
              'overlap_f1': 100*overlap_f}
    return result


# ------------ START: This part is for nbest predictions with confidence ---------- #

def eval_with_nbest_preds(nbest_file, gold_file):
    """ To use this part, check nbest output format of huggingface qa script """
    best_threshold,_ = find_best_threshold(nbest_file, gold_file)
    nbest_preds = read_nbest_pred(nbest_file)
    golds = read_gold(gold_file)
    preds = apply_threshold_nbest(best_threshold, nbest_preds)
    return multi_span_evaluate(preds, golds)


def check_overlap(offsets1, offsets2):
    if (offsets1[0]<=offsets2[0] and offsets1[1]>=offsets2[0]) or\
       (offsets1[0]>=offsets2[0] and offsets1[0]<=offsets2[1]):
        return True
    return False

def remove_overlapped_pred(pred):
    new_pred = [pred[0]]
    for p in pred[1:]:
        no_overlap = True
        for g in new_pred:
            if check_overlap(p['offsets'],g['offsets']):
                no_overlap = False
        if no_overlap:
            new_pred.append(p)
    return new_pred

def read_nbest_pred(nbest_pred_file):
    with open(nbest_pred_file) as f:
        nbest_pred = json.load(f)
    # Remove overlapped pred and normalize the answer text
    for k,v in nbest_pred.items():
        new_v = remove_overlapped_pred(v)
        for vv in new_v:
            vv['text'] = normalize_answer(vv['text'])
        nbest_pred[k] = new_v
    return nbest_pred

def apply_threshold_nbest(threshold, nbest_preds):
    preds = {}
    for k,v in nbest_preds.items():
        other_pred = filter(lambda x: x['probability']>= threshold, nbest_preds[k][1:]) # other preds except the first one
        if nbest_preds[k][0]['text'] != '': # only apply to the has_answer examples
            preds[k] = list(set([nbest_preds[k][0]['text']] + list(map(lambda x: x['text'], other_pred))))
        else:
            preds[k] = ['']
    return preds

def threshold2f1(threshold, golds, nbest_preds):
    preds = apply_threshold_nbest(threshold, nbest_preds)
    _,_,f1 = compute_scores(golds, preds, eval_type='em')
    return f1

def find_best_threshold(nbest_dev_file, gold_dev_file):
    golds = read_gold(gold_dev_file)
    nbest_preds = read_nbest_pred(nbest_dev_file)
    probs = list(map(lambda x:x[0]['probability'], nbest_preds.values()))
    sorted_probs = sorted(probs, reverse=True)
    # search probs in prob list and find the best threshold
    best_threshold = 0.5
    best_f1 = threshold2f1(0.5, golds, nbest_preds)
    for prob in sorted_probs:
        if prob > 0.5:
            continue
        cur_f1 = threshold2f1(prob, golds, nbest_preds)
        if cur_f1 > best_f1:
            best_f1 = cur_f1
            best_threshold = prob
    return best_threshold, best_f1
# ------------ END: This part is for nbest predictions with confidence ---------- #


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', default="", type=str)
    parser.add_argument('--gold_file', default="", type=str)
    args = parser.parse_args()
    result = multi_span_evaluate_from_file(args.pred_file, args.gold_file)
    for k,v in result.items():
        print(f"{k}: {v}")
