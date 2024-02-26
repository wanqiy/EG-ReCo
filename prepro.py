# -*- coding:utf-8 -*-
from collections import defaultdict
import json
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import ujson as json


docred_rel2id = json.load(open('dataset/meta/rel2id.json', 'r'))
docred_ent2id = {'NA': 0, 'ORG': 1, 'LOC': 2, 'NUM': 3, 'TIME': 4, 'MISC': 5, 'PER': 6}

#dreeam‘s prepro
def add_entity_markers(sample, tokenizer, entity_start, entity_end):
    ''' add entity marker (*) at the end and beginning of entities. '''

    sents = []
    sent_map = []
    sent_pos = []

    sent_start = 0
    for i_s, sent in enumerate(sample['sents']):
        # add * marks to the beginning and end of entities
        new_map = {}

        for i_t, token in enumerate(sent):
            tokens_wordpiece = tokenizer.tokenize(token)
            if (i_s, i_t) in entity_start:
                tokens_wordpiece = ["*"] + tokens_wordpiece
            if (i_s, i_t) in entity_end:
                tokens_wordpiece = tokens_wordpiece + ["*"]
            new_map[i_t] = len(sents)
            sents.extend(tokens_wordpiece)

        sent_end = len(sents)
        # [sent_start, sent_end)
        sent_pos.append((sent_start, sent_end,))
        sent_start = sent_end

        # update the start/end position of each token.
        new_map[i_t + 1] = len(sents)
        sent_map.append(new_map)

    return sents, sent_map, sent_pos


# three's make_path
def extract_path(data, keep_sent_order):
    sents = data["sents"]
    nodes = [[] for _ in range(len(data['sents']))]
    e2e_sent = defaultdict(dict)

    # create mention's list for each sentence
    for ns_no, ns in enumerate(data['vertexSet']):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(ns_no)

    for sent_id in range(len(sents)):
        for n1 in nodes[sent_id]:
            for n2 in nodes[sent_id]:
                if n1 == n2:
                    continue
                if n2 not in e2e_sent[n1]:
                    e2e_sent[n1][n2] = set()
                e2e_sent[n1][n2].add(sent_id)

    # 2-hop Path
    path_two = defaultdict(dict)
    entityNum = len(data['vertexSet'])
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                    continue
                for s1 in e2e_sent[n1][n3]:
                    for s2 in e2e_sent[n3][n2]:
                        if s1 == s2:
                            continue
                        if n2 not in path_two[n1]:
                            path_two[n1][n2] = []
                        cand_sents = [s1, s2]
                        if keep_sent_order == True:
                            cand_sents.sort()
                        path_two[n1][n2].append((cand_sents, n3))

    # 3-hop Path
    path_three = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue
                if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                    for cand1 in e2e_sent[n1][n3]:
                        for cand2 in path_two[n3][n2]:
                            if cand1 in cand2[0]:
                                continue
                            if cand2[1] == n1:
                                continue
                            if n2 not in path_three[n1]:
                                path_three[n1][n2] = []
                            cand_sents = [cand1] + cand2[0]
                            if keep_sent_order:
                                cand_sents.sort()
                            path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))

    # Consecutive Path
    consecutive = defaultdict(dict)
    for h in range(entityNum):
        for t in range(h + 1, entityNum):
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    gap = abs(n1['sent_id'] - n2['sent_id'])
                    if gap > 2:
                        continue
                    if t not in consecutive[h]:
                        consecutive[h][t] = []
                        consecutive[t][h] = []
                    if n1['sent_id'] < n2['sent_id']:
                        beg, end = n1['sent_id'], n2['sent_id']
                    else:
                        beg, end = n2['sent_id'], n1['sent_id']

                    consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                    consecutive[t][h].append([[i for i in range(beg, end + 1)]])

    # Merge
    merge = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n2 in path_two[n1]:
                merge[n1][n2] = path_two[n1][n2]
            if n2 in path_three[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += path_three[n1][n2]
                else:
                    merge[n1][n2] = path_three[n1][n2]

            if n2 in consecutive[n1]:
                if n2 in merge[n1]:
                    merge[n1][n2] += consecutive[n1][n2]
                else:
                    merge[n1][n2] = consecutive[n1][n2]

    # Default Path
    for h in range(len(data['vertexSet'])):
        for t in range(len(data['vertexSet'])):
            if h == t:
                continue
            if t in merge[h]:
                continue
            merge[h][t] = []
            for n1 in data['vertexSet'][h]:
                for n2 in data['vertexSet'][t]:
                    cand_sents = [n1['sent_id'], n2['sent_id']]
                    if keep_sent_order:
                        cand_sents.sort()
                    merge[h][t].append([cand_sents])

    # Remove redundency
    tp_set = set()
    for n1 in merge.keys():
        for n2 in merge[n1].keys():
            hash_set = set()
            new_list = []
            for t in merge[n1][n2]:
                if tuple(t[0]) not in hash_set:
                    hash_set.add(tuple(t[0]))
                    new_list.append(t[0])
            new_list = list(set(item for sublist in new_list for item in sublist))
            merge[n1][n2] = new_list

    return merge


def get_pseudo_features(raw_feature: dict, pred_rels: list, entities: list, sent_map: dict, offset: int,
                        tokenizer=None):
    ''' Construct pseudo documents from predictions.'''

    pos_samples = 0
    neg_samples = 0

    sent_grps = []
    pseudo_features = []

    for pred_rel in pred_rels:
        curr_sents = pred_rel["evidence"]  # evidence sentence
        if len(curr_sents) == 0:
            continue

        # check if head/tail entity presents in evidence. if not, append sentence containing the first mention of head/tail into curr_sents
        head_sents = sorted([m["sent_id"] for m in entities[pred_rel["h_idx"]]])
        tail_sents = sorted([m["sent_id"] for m in entities[pred_rel["t_idx"]]])

        if len(set(head_sents) & set(curr_sents)) == 0:
            curr_sents.append(head_sents[0])
        if len(set(tail_sents) & set(curr_sents)) == 0:
            curr_sents.append(tail_sents[0])

        curr_sents = sorted(set(curr_sents))
        if curr_sents in sent_grps:  # skip if such sentence group has already been created
            continue
        sent_grps.append(curr_sents)

        # new sentence masks and input ids
        old_sent_pos = [raw_feature["sent_pos"][i] for i in curr_sents]
        new_input_ids_each = [raw_feature["input_ids"][s[0] + offset:s[1] + offset] for s in old_sent_pos]
        new_input_ids = sum(new_input_ids_each, [])
        new_input_ids = tokenizer.build_inputs_with_special_tokens(new_input_ids)

        new_sent_pos = []

        prev_len = 0
        for sent in old_sent_pos:
            curr_sent_pos = (prev_len, prev_len + sent[1] - sent[0])
            new_sent_pos.append(curr_sent_pos)
            prev_len += sent[1] - sent[0]

        # iterate through all entities, keep only entities with mention in curr_sents.

        # obtain entity positions w.r.t whole document
        curr_entities = []
        ent_new2old = {}  # head/tail of a relation should be selected
        new_entity_pos = []

        for i, entity in enumerate(entities):
            curr = []
            curr_pos = []
            for mention in entity:
                if mention["sent_id"] in curr_sents:
                    curr.append(mention)
                    prev_len = new_sent_pos[curr_sents.index(mention["sent_id"])][0]
                    pos = [sent_map[mention["sent_id"]][pos] - sent_map[mention["sent_id"]][0] + prev_len for pos in
                           mention['pos']]
                    curr_pos.append(pos)

            if curr != []:
                curr_entities.append(curr)
                new_entity_pos.append(curr_pos)
                ent_new2old[len(ent_new2old)] = i  # update dictionary

        # iterate through all entities to obtain all entity pairs
        new_hts = []
        new_labels = []
        for h in range(len(curr_entities)):
            for t in range(len(curr_entities)):
                if h != t:
                    new_hts.append([h, t])
                    old_h, old_t = ent_new2old[h], ent_new2old[t]
                    curr_label = raw_feature["labels"][raw_feature["hts"].index([old_h, old_t])]
                    new_labels.append(curr_label)

                    neg_samples += curr_label[0]
                    pos_samples += 1 - curr_label[0]

        pseudo_feature = {'input_ids': new_input_ids,
                          'entity_pos': new_entity_pos,
                          'labels': new_labels,
                          'hts': new_hts,
                          'sent_pos': new_sent_pos,
                          'sent_labels': None,
                          'title': raw_feature['title'],
                          'entity_map': ent_new2old,
                          }
        pseudo_features.append(pseudo_feature)

    return pseudo_features, pos_samples, neg_samples


def read_docred(file_in,
                tokenizer,
                transformer_type="bert",
                max_seq_length=1024,
                teacher_sig_path="",
                single_results=None,
                keep_sent_order=False,
                tag=""):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    global attns

    if file_in == "":
        return None

    with open(file_in, "r") as fh:
        data = json.load(fh)

    if teacher_sig_path != "":  # load logits
        basename = os.path.splitext(os.path.basename(file_in))[0]
        attns_file = os.path.join(teacher_sig_path, f"{basename}.attns")
        with open(attns_file,'rb') as file:
        # attns = pickle.load(open(attns_file, 'rb'))
            attns = pickle.load(file)

    if single_results != None:
        # reorder predictions as relations by title
        pred_pos_samples = 0
        pred_neg_samples = 0
        pred_rels = single_results
        title2preds = {}
        for pred_rel in pred_rels:
            if pred_rel["title"] in title2preds:
                title2preds[pred_rel["title"]].append(pred_rel)
            else:
                title2preds[pred_rel["title"]] = [pred_rel]

    for doc_id, data in tqdm(enumerate(data), desc="Loading examples"):

        merge = extract_path(data, keep_sent_order)
        sample = data
        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        # record entities
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))

        sents, sent_map, sent_pos = add_entity_markers(sample, tokenizer, entity_start, entity_end)

        # training triples with positive examples (entity pairs with labels)
        train_triple = {}

        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])

                # update training triples
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        # entity start, end position
        entity_pos = []

        for e in entities:
            entity_pos.append([])
            assert len(e) != 0
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                label = m["type"]
                entity_pos[-1].append((start, end,))

        relations, hts, sent_labels = [], [], []

        for h, t in train_triple.keys():  # for every entity pair with gold relation
            relation = [0] * len(docred_rel2id)
            sent_evi = [0] * len(sent_pos)
            if file_in == "dataset/docred/test.json":
                for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
                    relation[mention["relation"]] = 1
                    sent_ids_list = merge[h][t]
                    for i in sent_ids_list:
                        sent_evi[i] += 1
            else:
                for mention in train_triple[h, t]:  # for each relation mention with head h and tail t
                    relation[mention["relation"]] = 1
                    for i in mention["evidence"]:
                        sent_evi[i] += 1



            relations.append(relation)
            hts.append([h, t])
            sent_labels.append(sent_evi)
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                # all entity pairs that do not have relation are treated as negative samples
                if h != t and [h, t] not in hts:  # and [t, h] not in hts:
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    sent_evi = [0] * len(sent_pos)
                    relations.append(relation)

                    hts.append([h, t])
                    sent_labels.append(sent_evi)
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)
        assert len(sents) < max_seq_length
        sents = sents[:max_seq_length - 2]  # truncate, -2 for [CLS] and [SEP]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        feature = [{'input_ids': input_ids,
                    'entity_pos': entity_pos,
                    'labels': relations,
                    'hts': hts,
                    'sent_pos': sent_pos,
                    'sent_labels': sent_labels,
                    'doc_id':doc_id,
                    'title': sample['title'],
                    }]

        if teacher_sig_path != '':  # add evidence distributions from the teacher model
            feature[0]['attns'] = attns[doc_id][:, :len(input_ids)]

        if single_results != None:  # get pseudo documents from predictions of the single run
            offset = 1 if transformer_type in ["bert", "roberta"] else 0
            if sample["title"] in title2preds:
                feature, pos_sample, neg_sample, = get_pseudo_features(feature[0], title2preds[sample["title"]],
                                                                       entities, sent_map, offset, tokenizer)
                pred_pos_samples += pos_sample
                pred_neg_samples += neg_sample

        i_line += len(feature)
        features.extend(feature)

    print("# of documents {}.".format(i_line))
    if single_results != None:
        print("# of positive examples {}.".format(pred_pos_samples))
        print("# of negative examples {}.".format(pred_neg_samples))

    else:
        print("# of positive examples {}.".format(pos_samples))
        print("# of negative examples {}.".format(neg_samples))

    return features


