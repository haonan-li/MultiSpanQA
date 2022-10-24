# Script for MultiSpanQA evaluation
import re
import json
import string
import difflib
import warnings
import numpy as np
from typing import Dict, List, Set


def get_entities(label, token):
    prev_tag = 'O'
    begin_offset = 0
    chunks = []

    # check no ent
    if isinstance(label[0], list):
        for i, s in enumerate(label):
            if len(set(s)) == 1:
                chunks.append(('O', -i, -i))
    # for nested list
    if any(isinstance(s, list) for s in label):
        label = [item for sublist in label for item in sublist + ['O']]
    if any(isinstance(s, list) for s in token):
        token = [item for sublist in token for item in sublist + ['O']]

    for i, chunk in enumerate(label + ['O']):
        if chunk not in ["O", "B", "I"]:
            warnings.warn('{} seems not to be IOB tag.'.format(chunk))
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):
            chunks.append((' '.join(token[begin_offset:i]), begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
    return chunks


def end_of_chunk(prev_tag: str, tag: str):
    """Determine if we are at the end of an answer chunk.

    :param prev_tag: previous tag
    :param tag: current tag
    """
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


def start_of_chunk(prev_tag: str, tag: str):
    """Determine if we are at the start of an answer chunk.

    :param prev_tag: previous tag
    :param tag: current tag
    """
    chunk_start = False
    if tag == 'B':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True
    return chunk_start


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace.

    :param s: input string
    """
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


def compute_scores(golds: Dict[Set[str]], preds: Dict[Set[str]], eval_type: str = 'em', average: str = 'micro'):
    """Compute precision, recall and exact match (or f1) metrics.

    :param golds: dictionary of gold XX
    :param preds: dictionary of predictions
    :param eval_type: Evaluation type. Can be either "em" or "overlap".
    """
    nb_gold = 0
    nb_pred = 0
    nb_correct = 0
    nb_correct_p = 0
    nb_correct_r = 0
    for keys in list(golds.keys()):
        gold = golds[keys]
        pred = preds[keys]
        nb_gold += max(len(gold), 1)
        nb_pred += max(len(pred), 1)
        if eval_type == 'em':
            if len(gold) == 0 and len(pred) == 0:
                # Exact match no answer case
                nb_correct += 1
            else:
                # Exact match comparison between two sets
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

    return p, r, f


def count_overlap(gold: set, pred: set):
    """Count the overlap of the gold answer and the predicted answer.

    :param gold: Set of gold answers
    :param pred: Set of predicted answers
    """
    # Correct no answer prediction
    if len(gold) == 0 and (len(pred) == 0 or pred == {""}):
        return 1, 1

    # Incorrect no answer prediction
    elif len(gold) == 0 or (len(pred) == 0 or pred == {""}):
        return 0, 0

    # NOTE: Since it is possible to return multiple spans it is not clear which spans from pred should be compared to
    #       each span in gold. So all are compared and the highest precision and recall are taken.
    p_scores = np.zeros((len(gold), len(pred)))
    r_scores = np.zeros((len(gold), len(pred)))
    for i, gold_str in enumerate(gold):
        for j, pred_str in enumerate(pred):
            seq_matcher = difflib.SequenceMatcher(None, gold_str, pred_str)
            _, _, longest_len = seq_matcher.find_longest_match(0, len(gold_str), 0, len(pred_str))
            p_scores[i][j] = longest_len/len(pred_str) if longest_len > 0 else 0
            r_scores[i][j] = longest_len/len(gold_str) if longest_len > 0 else 0

    p_score = sum(np.max(p_scores, axis=0))
    r_score = sum(np.max(r_scores, axis=1))

    return p_score, r_score


def read_gold(gold_file):
    """Read the gold file

    :param gold_file: file path to the file with the golden answers
    """
    with open(gold_file) as f:
        data = json.load(f)['data']
        golds = {}
        for piece in data:
            golds[piece['id']] = set(map(lambda x: x[0], get_entities(piece['label'], piece['context'])))
    return golds


def read_pred(pred_file):
    """Read the prediction file

    :param pred_file: file path to the prediction file
    """
    with open(pred_file) as f:
        preds = json.load(f)
    return preds


def multi_span_evaluate_from_file(pred_file, gold_file):
    preds = read_pred(pred_file)
    golds = read_gold(gold_file)
    result = multi_span_evaluate(preds, golds)
    return result


def multi_span_evaluate(preds: Dict[List[str]], golds: Dict[List[str]]):
    assert len(preds) == len(golds)
    assert preds.keys() == golds.keys()

    # Normalize the answer
    for k, v in golds.items():
        golds[k] = set(map(lambda x: normalize_answer(x), v))
    for k, v in preds.items():
        preds[k] = set(map(lambda x: normalize_answer(x), v))

    # Evaluate
    em_p, em_r, em_f = compute_scores(golds, preds, eval_type='em')  # type: ignore
    overlap_p, overlap_r, overlap_f = compute_scores(golds, preds, eval_type='overlap')  # type: ignore
    result = {'exact_match_precision': 100 * em_p,
              'exact_match_recall': 100 * em_r,
              'exact_match_f1': 100 * em_f,
              'overlap_precision': 100 * overlap_p,
              'overlap_recall': 100 * overlap_r,
              'overlap_f1': 100 * overlap_f}
    return result


# ------------ START: This part is for nbest predictions with confidence ---------- #

def eval_with_nbest_preds(nbest_file, gold_file):
    """ To use this part, check nbest output format of huggingface qa script """
    best_threshold, _ = find_best_threshold(nbest_file, gold_file)
    nbest_preds = read_nbest_pred(nbest_file)
    golds = read_gold(gold_file)
    preds = apply_threshold_nbest(best_threshold, nbest_preds)
    return multi_span_evaluate(preds, golds)


def check_overlap(offsets1, offsets2):
    if (offsets1[0] <= offsets2[0] and offsets1[1] >= offsets2[0]) or\
       (offsets1[0] >= offsets2[0] and offsets1[0] <= offsets2[1]):
        return True
    return False


def remove_overlapped_pred(pred):
    new_pred = [pred[0]]
    for p in pred[1:]:
        no_overlap = True
        for g in new_pred:
            if check_overlap(p['offsets'], g['offsets']):
                no_overlap = False
        if no_overlap:
            new_pred.append(p)
    return new_pred


def read_nbest_pred(nbest_pred_file):
    with open(nbest_pred_file) as f:
        nbest_pred = json.load(f)
    # Remove overlapped pred and normalize the answer text
    for k, v in nbest_pred.items():
        new_v = remove_overlapped_pred(v)
        for vv in new_v:
            vv['text'] = normalize_answer(vv['text'])
        nbest_pred[k] = new_v
    return nbest_pred


def apply_threshold_nbest(threshold, nbest_preds):
    preds = {}
    for k, v in nbest_preds.items():
        other_pred = filter(lambda x: x['probability'] >= threshold, nbest_preds[k][1:])  # other preds except the first one
        if nbest_preds[k][0]['text'] != '':  # only apply to the has_answer examples
            preds[k] = list(set([nbest_preds[k][0]['text']] + list(map(lambda x: x['text'], other_pred))))
        else:
            preds[k] = ['']
    return preds


def threshold2f1(threshold, golds, nbest_preds):
    preds = apply_threshold_nbest(threshold, nbest_preds)
    _, _, f1 = compute_scores(golds, preds, eval_type='em')
    return f1


def find_best_threshold(nbest_dev_file, gold_dev_file):
    golds = read_gold(gold_dev_file)
    nbest_preds = read_nbest_pred(nbest_dev_file)
    probs = list(map(lambda x: x[0]['probability'], nbest_preds.values()))
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
    eval_result = multi_span_evaluate_from_file(args.pred_file, args.gold_file)
    for k, v in eval_result.items():
        print(f"{k}: {v}")
