# -*- coding: utf-8 -*-
# @Time : 2023/7/30 17:52
# model :
# @Author : YangHao

import torch
import torch.nn as nn
from opt_einsum import contract
import torch.nn.functional as F
import numpy as np
from long_seq import process_long_input
from losses import ATLoss
import  args



class Agent(nn.Module):
    def __init__(self, config, model, tokenizer,
                 emb_size=768, block_size=64, num_labels=-1,
                 max_sent_num=25,evi_thresh=0.2):
        '''
        Initialize the model.
        :model: Pretrained langage model encoder;
        :tokenizer: Tokenzier corresponding to the pretrained language model encoder;
        :emb_size: Dimension of embeddings for subject/object (head/tail) representations;
        :block_size: Number of blocks for grouped bilinear classification;
        :num_labels: Maximum number of relation labels for each entity pair;
        :max_sent_num: Maximum number of sentences for each document;
        :evi_thresh: Threshold for selecting evidence sentences.
        '''

        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.hidden_size = config.hidden_size

        self.loss_fnt = ATLoss()
        self.loss_fnt_evi = nn.KLDivLoss(reduction="batchmean")

        self.head_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, emb_size)
        self.state_extractor = nn.Linear(self.hidden_size * 2, emb_size)

        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.total_labels = config.num_labels
        self.max_sent_num = max_sent_num
        self.evi_thresh = evi_thresh


        self.epsilon = 1.0
        self.eps_min = 0.01
        self.eps_dec = 5e-7
        self.action_dim = 2
        self.batch_size = 1
        self.action_space = [i for i in range(self.action_dim)]

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


        self.alpha = 3e-5
        self.state_dim = emb_size
        self.fc1_dim = 256
        self.fc2_dim = 256
        self.gamma = 0.99
        tau = 0.005

        # # Assuming logits and s_action are PyTorch tensors
        # self.neg_log_prob = F.cross_entropy(logits, s_action)
        #
        # # Assuming neg_log_prob and s_value are PyTorch tensors
        # self.evi_loss = torch.mean(neg_log_prob * s_value)

        self.q_eval = DuelingDeepQNetwork(alpha=self.alpha, state_dim=self.state_dim, action_dim=self.action_dim,
                                          fc1_dim=self.fc1_dim, fc2_dim=self.fc2_dim)


    def encode(self, input_ids, attention_mask):
        '''
        Get the embedding of each token. For long document that has more than 512 tokens, split it into two overlapping chunks.
        Inputs:
            :input_ids: (batch_size, doc_len)
            :attention_mask: (batch_size, doc_len)
        Outputs:
            :sequence_output: (batch_size, doc_len, hidden_dim)
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
        '''
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        # process long documents.
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)

        return sequence_output, attention


    def get_hrt_rl(self, sequence_output, attention, entity_pos, hts, offset, sent_pos):
        '''
        Get head, tail, context embeddings from token embeddings.
        Inputs:
            :sequence_output: (batch_size, doc_len, hidden_dim) doc_len:document's number
            :attention: (batch_size, num_attn_heads, doc_len, doc_len)
            :entity_pos: list of list. Outer length = batch size, inner length = number of entities each batch.
            :hts: list of list. Outer length = batch size, inner length = number of combination of entity pairs each batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
            :sent_labels: list of list. Outer length = batch size, inner length = number of sentences each batch.
        Outputs:
            :hss: (num_ent_pairs_all_batches, emb_size)
            :tss: (num_ent_pairs_all_batches, emb_size)
            :rss: (num_ent_pairs_all_batches, emb_size)
            :ht_atts: (num_ent_pairs_all_batches, doc_len)
            :stence_emb: (num_ent_pairs_all_batches, len(sent_pos), emb_size)
            :rels_per_batch: list of length = batch size. Each entry represents the number of entity pairs of the batch.
        '''

        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        ht_atts = []
        stence_emb_all_batches = []

        for i in range(len(entity_pos)):  # for each batch
            entity_embs, entity_atts, selected_sent_emb = [], [], []
            curr_sent_pos = sent_pos[i]  # 当前批次中每个句子的起始和结束位置
            for mid,(start,end) in enumerate(curr_sent_pos):
                stence_emb = []
                if start + offset < c :
                    for start_i in range(start,end):
                        stence_emb.append(sequence_output[i,start_i+offset])

                    if len(stence_emb) > 0:
                        stence_emb = torch.logsumexp(torch.stack(stence_emb, dim=0), dim=0)
                    else:
                        stence_emb = torch.zeros(self.config.hidden_size).to(sequence_output)

                selected_sent_emb.append(stence_emb)

            selected_sent_emb = torch.stack(selected_sent_emb, dim=0)


            # obtain entity embedding from mention embeddings.
            for eid, e in enumerate(entity_pos[i]):  # for each entity
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for mid, (start, end) in enumerate(e):  # for every mention
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])

                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)

                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]



            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)

            # obtain subject/object (head/tail) embeddings from entity embeddings.
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            num_row = hs.size(0)

            # 扩展 sent 张量的维度，使其形状变成 (1, 6, 768)
            expanded_sent = selected_sent_emb.unsqueeze(0)

            # 使用 repeat 函数复制张量，使其形状变成 (12, 6, 768)
            stence_tensor = expanded_sent.repeat(num_row, 1, 1)


            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])

            ht_att = (h_att * t_att).mean(1)  # average over all heads
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            ht_atts.append(ht_att)

            # obtain local context embeddings.
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            stence_emb_all_batches.append(stence_tensor)

        rels_per_batch = [len(b) for b in hss]
        hss = torch.cat(hss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        tss = torch.cat(tss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        rss = torch.cat(rss, dim=0)  # (num_ent_pairs_all_batches, emb_size)
        ht_atts = torch.cat(ht_atts, dim=0)  # (num_ent_pairs_all_batches, max_doc_len)


        # stence_emb = torch.cat(stence_emb, dim=0)  # (num_ent_pairs_all_batches,len(sent_pos),emb_size)

        return hss, rss, tss, ht_atts, stence_emb_all_batches, rels_per_batch


    def get_css(self, stence_emb, pred_lables, rss):
        alpha = 0.9
        css = []
        css_all_emb = []
        for i in range(0,len(stence_emb)):
            pred_lable = pred_lables[i]
            css_embs = []
            for j in range(0,len(pred_lable)):
                css_emb = []
                for z in range(0,len(pred_lable[0])):
                    if pred_lable[j][z] !=0:
                        css_emb.append(stence_emb[i][j][z])
                if len(css_emb)>0:
                    css_emb = torch.logsumexp(torch.stack(css_emb,dim=0),dim=0)
                else:
                    css_emb = torch.zeros(self.config.hidden_size).to(self.device)

                css_embs.append(css_emb)
            css_embs = torch.stack(css_embs,dim=0)
            css.append(css_embs)

        css = torch.cat(css,dim=0)

        fused_info = alpha * rss + (1 - alpha) * css
        return fused_info,css


    def forward_rel_rl(self, hs, ts, cs):
        '''
        Forward computation for RE.
        Inputs:
            :hs: (num_ent_pairs_all_batches, emb_size)
            :ts: (num_ent_pairs_all_batches, emb_size)
            :rs: (num_ent_pairs_all_batches, emb_size)
        Outputs:
            :logits: (num_ent_pairs_all_batches, num_rel_labels)
        '''

        hss = torch.tanh(self.head_extractor(torch.cat([hs, cs], dim=-1)))
        tss = torch.tanh(self.tail_extractor(torch.cat([ts, cs], dim=-1)))

        # hs = torch.relu(hss)
        # ts = torch.relu(tss)

        # split into several groups.
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)

        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        return logits

    def choose_action(self, state, stence_emb, isTrain=True):
        state = state.to(self.device)
        q_vals = self.q_eval.forward(state)
        action = torch.argmax(q_vals).item()
        softmax_q_vals = torch.softmax(q_vals,dim=0)
        # neg_log_prog = F.cross_entropy(q_vals,action,reduction='mean')

        # if (np.random.random() < self.epsilon) and isTrain:
        #     action = np.random.choice(self.action_space)

        return action, state, q_vals,softmax_q_vals

    def step(self,sent_labels,ht_state,stence_emb,sent_pos):
        # 获取矩阵的行数和列数
        max_sent_num = max([len(sent) for sent in sent_pos])
        s_num = 0
        p_num = 0
        t_num = 0
        pred_lables = []
        pred_qs = []
        neg_log_probs = []
        for i in range(0,len(sent_labels)):
            num_rows = len(sent_labels[i])
            num_cols = len(sent_labels[i][0])
            sent_label = sent_labels[i]

            # 初始化与 sent_labels 大小相同的全零矩阵
            pred_lable = torch.zeros(num_rows, max_sent_num, dtype=torch.int64)
            pred_q = torch.zeros(num_rows, max_sent_num, dtype=torch.int64)
            neg_log_prob = torch.zeros(num_rows, max_sent_num, dtype=torch.float64)

            for s_i in range(0,stence_emb[i].size(0)):
                state_s = ht_state[s_num]
                s_num += 1
                for j in range(0,stence_emb[i].size(1)):
                    if sent_label[s_i][j] != 0:
                        t_num +=1
                        state_m = state_s
                        state = self.state_extractor(torch.cat([state_m, stence_emb[i][s_i][j]], dim=-1))

                        action, state_d, q_val,softmax_q = self.choose_action(state, stence_emb[i][s_i][j], isTrain=True)
                        if action == 1:
                            state_s = state_d
                            pred_lable[s_i][j] = 1
                            p_num +=1
                        else:
                            state_s = state_m
                        pred_q[s_i][j] = softmax_q[1].item()
                        # q_val_2d = q_val.reshape(1,2)
                        # target_action = torch.tensor(1).unsqueeze(0)
                        # neg_log_prob[s_i][j] = F.cross_entropy(input=q_val_2d, target=target_action.to(self.device), reduction='mean')
                        neg_log_prob[s_i][j] = F.cross_entropy(input=q_val, target=torch.tensor(1).to(self.device,dtype=torch.long),
                                                               reduction='mean')
            pred_lables.append(pred_lable)
            pred_qs.append(pred_q)
            neg_log_probs.append(neg_log_prob)
            pred_lable_st = torch.cat(pred_lables,dim=0)
        # 返回下一个观测值、奖励、是否结束和其他信息
        return pred_lables, pred_qs,neg_log_probs,pred_lable_st

    def get_rewards(self, labels, logits):
        self.output = logits
        logits = torch.sigmoid(logits)
        logits = torch.clamp(logits, 1e-3, 1 - 1e-3)
        self.logits = logits

        log1 = -labels * torch.log(1-logits)
        log2 = -(1-labels) * torch.log(logits)

        log = log1 + log2

        self.log1 = log1
        self.log2 = log2
        rewards = torch.mean(log,dim=1)

        return rewards

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def get_value(self,reward,pred_lables):
        value_doc = []
        re_count  = 0
        for i in range(len(pred_lables)):
            pred_lable = pred_lables[i]
            pred_len  = len(pred_lable)
            value_sent = []
            for j in range(pred_len):
                rewards = [0.0 for x in range(len(pred_lable[j]))]
                rewards[len(rewards)-1] = reward[re_count]
                values = [0.0 for x in range(len(pred_lable[j]))]
                running_add = 0
                re_count += 1
                for r_i in reversed(range(0,len(rewards))):
                    running_add = running_add * self.gamma + rewards[r_i]
                    values[r_i] = running_add
                value_sent.append(values)
            value_doc.append(value_sent)
        return  value_doc

    def get_loss_rl(self,neg,value):
        neg_st = torch.cat(neg, dim=0)
        tensor_st = torch.cat([torch.tensor(v) for v in value], dim=0)
        # neg_st_q = torch.clamp(neg_st,1e-3,1-(1e-3))
        one_e_minus_3 = torch.tensor(1e-3, dtype=torch.float64)
        neg_st_q = torch.where(neg_st == 0, one_e_minus_3, neg_st)
        evi_loss= torch.mean(neg_st_q * tensor_st)
        # evi_loss.requires_grad_(True)
        return evi_loss

    def forward_evi(self, doc_attn, sent_pos, batch_rel, offset,pred_lables,sent_labels):
        '''
        Forward computation for ER.
        Inputs:
            :doc_attn: (num_ent_pairs_all_batches, doc_len), attention weight of each token for computing localized context pooling.
            :sent_pos: list of list. The outer length = batch size. The inner list contains (start, end) position of each sentence in each batch.
            :batch_rel: list of length = batch size. Each entry represents the number of entity pairs of the batch.
            :offset: 1 for bert and roberta. Offset caused by [CLS] token.
        Outputs:
            :s_attn:  (num_ent_pairs_all_batches, max_sent_all_batch), sentence-level evidence distribution of each entity pair.
        '''

        w1 = 0.9
        max_sent_num = max([len(sent) for sent in sent_pos])
        rel_sent_attn = []
        for i in range(len(sent_pos)):  # for each batch
            # the relation ids corresponds to document in batch i is [sum(batch_rel[:i]), sum(batch_rel[:i+1]))
            curr_attn = doc_attn[sum(batch_rel[:i]):sum(batch_rel[:i + 1])]
            curr_sent_pos = [torch.arange(s[0], s[1]).to(curr_attn.device) + offset for s in sent_pos[i]]  # + offset

            curr_attn_per_sent = [curr_attn.index_select(-1, sent) for sent in curr_sent_pos]
            curr_attn_per_sent += [torch.zeros_like(curr_attn_per_sent[0])] * (max_sent_num - len(curr_attn_per_sent))
            sum_attn = torch.stack([attn.sum(dim=-1) for attn in curr_attn_per_sent],
                                   dim=-1)  # sum across those attentions
            rel_sent_attn.append(sum_attn)

        pred_lable = torch.cat(pred_lables,dim=0).to(self.device)
        s_attn_evi = torch.cat(rel_sent_attn, dim=0).to(self.device)
        s_attn = w1 * s_attn_evi + (1 - w1) * pred_lable * s_attn_evi

        if sent_labels != [] and None not in sent_labels:
            sent_labels_tensor = []
            for sent_label in sent_labels:
                sent_label = np.array(sent_label)
                sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent_num - sent_label.shape[1]))))
            sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
        else:
            sent_labels_tensor = None

        return s_attn,sent_labels_tensor

    # def sent_lable_tensor(self,sent_labels):
    #
    #     if sent_labels != [] and None not in sent_labels:
    #         sent_labels_tensor = []
    #         for sent_label in sent_labels:
    #             sent_label = np.array(sent_label)
    #             sent_labels_tensor.append(np.pad(sent_label, ((0, 0), (0, max_sent - sent_label.shape[1]))))
    #         sent_labels_tensor = torch.from_numpy(np.concatenate(sent_labels_tensor, axis=0))
    #     else:
    #         sent_labels_tensor = None
    #
    #         return sent_labels_tensor

    def forward(self,
                   input_ids=None,
                   attention_mask=None,
                   labels=None,  # relation labels
                   entity_pos=None,
                   hts=None,  # entity pairs
                   sent_pos=None,
                   sent_labels=None,  # evidence labels (0/1)
                   teacher_attns=None,  # evidence distribution from teacher model
                   doc_id = None,
                   title = None,
                   tag="train",
                   ):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        output = {}

        sequence_output, attention = self.encode(input_ids, attention_mask)

        hss, rss, tss, doc_attn, stence_emb, batch_rel = self.get_hrt_rl(sequence_output, attention,
                                                                                          entity_pos, hts, offset, sent_pos)

        ht_state = torch.mean(torch.stack([hss, tss], dim=0), dim=0)

        pred_lables, pred_q, neg_log, pred_lables_st = self.step(sent_labels,ht_state,stence_emb,sent_pos)

        # css,s_a = self.get_css(stence_emb,pred_lables,rss)

        logits = self.forward_rel_rl(hss, tss, rss)

        output["rel_pred"] = self.loss_fnt.get_label(logits, num_labels=self.num_labels)

        # if sent_labels != None:  # human-annotated evidence available
        #
        #     output["evi_pred"] = F.pad(pred_lables_st == 1, (0, self.max_sent_num - pred_lables_st.shape[-1]))
        if sent_labels != None: # human-annotated evidence available

            s_attn,sent_labels_tensor = self.forward_evi(doc_attn, sent_pos, batch_rel, offset,pred_lables,sent_labels)
            output["evi_pred"] = F.pad(s_attn > self.evi_thresh, (0, self.max_sent_num - s_attn.shape[-1]))

        if tag in ["test", "dev"]:  # testing
            scores_topk = self.loss_fnt.get_score(logits, self.num_labels)
            output["scores"] = scores_topk[0]
            output["topks"] = scores_topk[1]

        if tag == "infer":  # teacher model inference
            output["attns"] = doc_attn.split(batch_rel)

        else:  # training
            # relation extraction loss
            loss = self.loss_fnt(logits.float(), labels.float())
            output["loss"] = {"rel_loss": loss.to(sequence_output)}
            # reward = self.get_rewards(labels, logits)
            # values = self.get_value(reward,pred_lables)
            # evi_loss =self.get_loss_rl(neg_log,values)
            # output["loss"]["evi_loss"] = evi_loss.to(sequence_output)
            if sent_labels != None:  # supervised training with human evidence

                # idx_used = torch.nonzero(labels[:, 1:].sum(dim=-1)).view(-1)
                # # evidence retrieval loss (kldiv loss)
                # pred_q = pred_q[idx_used]
                # sent_labels = sent_labels[idx_used]
                # norm_s_labels = sent_labels / (sent_labels.sum(dim=-1, keepdim=True) + 1e-30)
                # norm_s_labels[norm_s_labels == 0] = 1e-30
                # pred_q[pred_q == 0] = 1e-30
                #
                # evi_loss = self.loss_fnt_evi(pred_q.log(), norm_s_labels)
                # evi_loss =  torch.mean(evi_loss * reward)
                # output["loss"]["evi_loss"] = evi_loss.to(sequence_output)

                idx_used = torch.nonzero(labels[:,1:].sum(dim=-1)).view(-1)
                # evidence retrieval loss (kldiv loss)
                s_attn = s_attn[idx_used]
                sent_labels_t = sent_labels_tensor[idx_used]
                norm_s_labels = sent_labels_t/(sent_labels_t.sum(dim=-1, keepdim=True) + 1e-30)
                norm_s_labels[norm_s_labels == 0] = 1e-30
                s_attn[s_attn == 0] = 1e-30
                norm_s_labels = norm_s_labels.to(sequence_output)
                s_attn = s_attn.to(sequence_output)
                evi_loss = self.loss_fnt_evi(s_attn.log(), norm_s_labels)
                output["loss"]["evi_loss"] = evi_loss.to(sequence_output)

            elif teacher_attns != None:  # self training with teacher attention

                doc_attn[doc_attn == 0] = 1e-30
                teacher_attns[teacher_attns == 0] = 1e-30
                attn_loss = self.loss_fnt_evi(doc_attn.log(), teacher_attns)
                output["loss"]["attn_loss"] = attn_loss.to(sequence_output)

        return output

class DuelingDeepQNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(DuelingDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.V = nn.Linear(fc2_dim, 1)
        self.A = nn.Linear(fc2_dim, action_dim)


    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))

        V = self.V(x)
        A = self.A(x)
        Q = V + A - torch.mean(A, dim=-1, keepdim=True)

        return Q