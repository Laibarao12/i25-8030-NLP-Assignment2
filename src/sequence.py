"""Part 2: from-scratch BiLSTM sequence labelling, CRF/Viterbi and ablations."""
from __future__ import annotations
import copy, json, time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from .common import DATA, EMB, MODELS, FIGS, RES, seed_all, device, plot_curves,plot_confusion,save_json
from .annotation import POS_TAGS,NER_TAGS


class TaggedDataset(Dataset):
    def __init__(self,split,task,vocab):
        path=DATA/f'{task}_{split}.conll'
        if not path.exists():raise FileNotFoundError(f'{path} missing: run src.annotation first')
        lines=path.read_text('utf-8').splitlines();self.rows=[];one=[]
        for line in lines+['']:
            if line.strip():
                parts=line.split('\t')
                if len(parts)!=2:raise ValueError(f'Bad CoNLL line: {line}')
                one.append(parts)
            elif one:
                self.rows.append(([w for w,t in one],[t for w,t in one]));one=[]
        self.vocab=vocab; self.tag2id={t:i for i,t in enumerate(POS_TAGS if task=='pos' else NER_TAGS)}
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        ts,tags=self.rows[i]
        return ([self.vocab.get(t.lower(),1) for t in ts], [self.tag2id[t] for t in tags],ts)


def collate(batch):
    seqs,tags,texts=zip(*batch);lengths=torch.tensor([len(s) for s in seqs],dtype=torch.long)
    maxlen=int(lengths.max());ids=torch.zeros(len(batch),maxlen,dtype=torch.long)
    yy=torch.zeros(len(batch),maxlen,dtype=torch.long)
    mask=torch.arange(maxlen)[None,:]<lengths[:,None]
    for i,(x,y) in enumerate(zip(seqs,tags)):
        ids[i,:len(x)]=torch.tensor(x); yy[i,:len(y)]=torch.tensor(y)
    return ids,yy,mask,lengths,list(texts)


class LinearChainCRF(nn.Module):
    """Learned start/end/tag transitions. Exact forward normalizer and Viterbi MAP."""
    def __init__(self,n):
        super().__init__();self.n=n
        self.transitions=nn.Parameter(torch.empty(n,n))  # [previous,current]
        self.start=nn.Parameter(torch.empty(n));self.end=nn.Parameter(torch.empty(n))
        nn.init.uniform_(self.transitions,-.1,.1)
        nn.init.uniform_(self.start,-.1,.1);nn.init.uniform_(self.end,-.1,.1)
    def log_partition(self,e,mask):
        score=self.start+e[:,0]
        for t in range(1,e.size(1)):
            nxt=torch.logsumexp(score[:,:,None]+self.transitions[None,:,:]+e[:,t,None,:],dim=1)
            score=torch.where(mask[:,t,None],nxt,score)
        return torch.logsumexp(score+self.end,dim=1)
    def gold_score(self,e,y,mask):
        B,T=y.shape
        score=self.start[y[:,0]]+e[:,0,:].gather(1,y[:,0,None]).squeeze(1)
        for t in range(1,T):
            emit=e[:,t,:].gather(1,y[:,t,None]).squeeze(1)
            link=self.transitions[y[:,t-1],y[:,t]]
            score+=(emit+link)*mask[:,t]
        lengths=mask.long().sum(1)-1
        last=y.gather(1,lengths[:,None]).squeeze(1)
        return score+self.end[last]
    def forward(self,emissions,tags,mask):
        return (self.log_partition(emissions,mask)-self.gold_score(emissions,tags,mask)).mean()
    def viterbi(self,e,mask):
        score=self.start+e[:,0]
        bps=[]
        for t in range(1,e.size(1)):
            poss=score[:,:,None]+self.transitions[None,:,:]
            best,back=poss.max(dim=1)
            nxt=best+e[:,t]
            score=torch.where(mask[:,t,None],nxt,score)
            bps.append(back)
        score=score+self.end
        seqs=[]
        for i,L in enumerate(mask.sum(1).tolist()):
            cur=int(score[i].argmax());path=[cur]
            for t in range(int(L)-2,-1,-1):
                cur=int(bps[t][i,cur]);path.append(cur)
            seqs.append(path[::-1])
        return seqs


class BiLSTMTagger(nn.Module):
    def __init__(self,vocab_size,num_tags,weights,freeze=False,bidirectional=True,
                 dropout=.5,random_init=False,use_crf=False):
        super().__init__();d=int(weights.shape[1]);hidden=64
        self.embedding=nn.Embedding(vocab_size,d,padding_idx=0)
        self.initial_embedding_stats={"source":"random" if random_init else "C3 Word2Vec"}
        if not random_init:
            pretrained=torch.from_numpy(weights.copy()).float()
            # Skip-gram vectors start much smaller than nn.Embedding's N(0,1)
            # weights. Match initial feature scale without replacing or
            # re-training the learned C3 vectors. Use one global factor, so
            # semantic directions and relative pairwise magnitudes are retained.
            if use_crf:
                valid=pretrained[2:]  # ignore PAD and UNK for scale estimate
                original_std=float(valid.std(unbiased=False).item()) if valid.numel() else 0.0
                if original_std < 1e-8:raise ValueError('C3 vectors are degenerate; inspect embeddings_w2v.npy')
                factor=min(100.0,1.0/original_std)
                pretrained.mul_(factor)
                self.initial_embedding_stats.update(raw_std=original_std,
                                                    global_scale=factor,
                                                    scaled_std=float(pretrained[2:].std(unbiased=False)))
            self.embedding.weight.data.copy_(pretrained)
        # Ensure padding stays zero for BOTH pretrained and random ablation.
        with torch.no_grad():self.embedding.weight[0].zero_()
        self.embedding.weight.requires_grad_(not freeze)
        self.dropout=nn.Dropout(dropout)
        self.lstm=nn.LSTM(d,hidden,num_layers=2,batch_first=True,
                          bidirectional=bidirectional,dropout=dropout)
        self.classifier=nn.Linear(hidden*(2 if bidirectional else 1),num_tags)
        self.crf=LinearChainCRF(num_tags) if use_crf else None
    def emissions(self,x,lengths):
        e=self.dropout(self.embedding(x))
        packed=pack_padded_sequence(e,lengths.cpu(),batch_first=True,enforce_sorted=False)
        encoded,_=self.lstm(packed)
        out,_=pad_packed_sequence(encoded,batch_first=True,total_length=x.shape[1])
        return self.classifier(self.dropout(out))
    def loss(self,emit,y,mask):
        if self.crf is not None:return self.crf(emit,y,mask)
        return F.cross_entropy(emit.reshape(-1,emit.shape[-1])[mask.reshape(-1)],y.reshape(-1)[mask.reshape(-1)])
    def predict(self,emit,mask):
        if self.crf is not None:return self.crf.viterbi(emit,mask)
        return [row[:int(L)].tolist() for row,L in zip(emit.argmax(-1),mask.sum(-1))]


def get_spans(tags):
    """Conlleval BIO chunk boundaries, treating illegal I-X as a new B-X."""
    chunks=[];start=None;typ=None
    for i,tag in enumerate(list(tags)+['O']):
        prefix,kind=(tag.split('-',1)+[None])[:2] if '-' in tag else (tag,None)
        if typ is not None and (prefix in ('O','B') or kind!=typ):
            chunks.append((typ,start,i));start=None;typ=None
        if prefix in ('B','I') and typ is None:
            start=i;typ=kind
    return set(chunks)


def conlleval_scores(true_seq,pred_seq):
    """Strict span-level CoNLL evaluation with matching type and token boundaries.
    Also writes results/conlleval_input.conll for optional external conlleval check.
    """
    tags=['PER','LOC','ORG','MISC'];stats={t:{'TP':0,'FP':0,'FN':0} for t in tags}
    fps=[];fns=[]
    for idx,(gold,pred) in enumerate(zip(true_seq,pred_seq)):
        ga=get_spans(gold);pa=get_spans(pred)
        for t,a,b in (pa & ga):stats[t]['TP']+=1
        for t,a,b in (pa-ga):stats[t]['FP']+=1;fps.append({'sentence_index':idx,'span':[a,b],'type':t,'error':'false_positive'})
        for t,a,b in (ga-pa):stats[t]['FN']+=1;fns.append({'sentence_index':idx,'span':[a,b],'type':t,'error':'false_negative'})
    output={}
    for t in tags+['overall']:
        counts=stats[t] if t!='overall' else {k:sum(v[k] for v in stats.values()) for k in ['TP','FP','FN']}
        tp,fp,fn=counts['TP'],counts['FP'],counts['FN']
        precision=tp/(tp+fp) if tp+fp else 0.
        recall=tp/(tp+fn) if tp+fn else 0.
        f1=2*precision*recall/(precision+recall) if precision+recall else 0.
        output[t]={**counts,'precision':precision,'recall':recall,'f1':f1}
    return output,fps,fns


def infer_dataset(model,loader,task,dev):
    model.eval();all_gold=[];all_pred=[];all_tokens=[];totalloss=0.;num=0
    tags=POS_TAGS if task=='pos' else NER_TAGS
    with torch.no_grad():
        for x,y,m,l,tokens in loader:
            x=x.to(dev);y=y.to(dev);m=m.to(dev)
            emit=model.emissions(x,l);totalloss+=float(model.loss(emit,y,m).item());num+=1
            pred=model.predict(emit,m)
            gold=[row[:int(L)].tolist() for row,L in zip(y,l)]
            all_gold.extend([[tags[i] for i in row] for row in gold])
            all_pred.extend([[tags[i] for i in row] for row in pred]);all_tokens.extend(tokens)
    if task=='pos':
        flat_g=[t for row in all_gold for t in row]; flat_p=[t for row in all_pred for t in row]
        metrics={'accuracy':float(accuracy_score(flat_g,flat_p)),
                 'macro_f1':float(f1_score(flat_g,flat_p,labels=POS_TAGS,average='macro',zero_division=0))}
        primary=metrics['macro_f1']
    else:
        metrics,_,_=conlleval_scores(all_gold,all_pred);primary=metrics['overall']['f1']
    return {'loss':totalloss/max(1,num),'primary_f1':primary,'metrics':metrics,
            'gold':all_gold,'pred':all_pred,'tokens':all_tokens}


def train_one(task,name,vocab,weights,freeze=False,bidir=True,dropout=.5,random_init=False,
              use_crf=False,max_epochs=30,batch_size=24):
    seed_all(42);dev=device()
    datasets={k:TaggedDataset(k,task,vocab) for k in ['train','val','test']}
    loaders={k:DataLoader(v,batch_size=batch_size,shuffle=(k=='train'),collate_fn=collate) for k,v in datasets.items()}
    if task=='ner':
        # Rebalance only TRAINING batches, not validation/test references.
        # Preserve whole BIO sentences and avoid generating synthetic labels.
        sentence_weights=[]
        for _,tags in datasets['train'].rows:
            spans=sum(x.startswith('B-') for x in tags)
            sentence_weights.append(min(8.0,1.0+4.0*spans))
        sampler=WeightedRandomSampler(sentence_weights,len(sentence_weights),replacement=True,
                                      generator=torch.Generator().manual_seed(42))
        loaders['train']=DataLoader(datasets['train'],batch_size=batch_size,
                                    sampler=sampler,collate_fn=collate)
    tags=POS_TAGS if task=='pos' else NER_TAGS
    model=BiLSTMTagger(len(vocab),len(tags),weights,freeze,bidir,dropout,random_init,use_crf).to(dev)
    opt=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=.001,weight_decay=.0001)
    class_weights=None
    if task=='ner':
        # Compute token weights from TRAIN annotations only; do not look at val/test.
        counts=Counter(t for _,sentence_tags in datasets['train'].rows for t in sentence_tags)
        background=max(1,counts['O'])
        vals=[1.0 if t=='O' else min(8.0,max(2.0,(background/max(1,counts[t]))**.5))
              for t in tags]
        class_weights=torch.tensor(vals,dtype=torch.float32,device=dev)
        print(name,'TRAIN-only NER class weights:',dict(zip(tags,[round(v,2) for v in vals])))
        print(name,'embedding initialization:',model.initial_embedding_stats)
    # Patience remains five validation-F1 non-improvements, but entity models
    # receive a minimum warm-up: their F1 is often zero in the early epochs.
    minimum_ner_epochs=12
    hist=[];best=-1.;bad=0;beststate=None;last_epoch=0
    for epoch in range(1,max_epochs+1):
        t0=time.perf_counter();model.train();sum_loss=0.;steps=0
        for x,y,m,l,_ in loaders['train']:
            x=x.to(dev);y=y.to(dev);m=m.to(dev)
            opt.zero_grad(set_to_none=True);em=model.emissions(x,l)
            loss=model.loss(em,y,m)
            if task=='ner':
                # Supplemental supervision combats the CRF's all-O collapse.
                # CRF + Viterbi remain the primary structured model and decoder.
                weighted_ce=F.cross_entropy(em[m],y[m],weight=class_weights)
                loss=(loss + 0.75*weighted_ce) if use_crf else weighted_ce
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=5.)
            opt.step();sum_loss+=float(loss.item());steps+=1
        va=infer_dataset(model,loaders['val'],task,dev)
        row={'epoch':epoch,'train_loss':sum_loss/max(steps,1),'val_loss':va['loss'],
             'val_f1':va['primary_f1'],'epoch_seconds':round(time.perf_counter()-t0,3)}
        hist.append(row);print(name,'epoch',epoch,'trainloss',round(row['train_loss'],4),'valF1',round(row['val_f1'],4))
        if va['primary_f1']>best+1e-7:
            best=va['primary_f1'];bad=0;beststate=copy.deepcopy(model.state_dict());last_epoch=epoch
        elif task!='ner' or epoch>minimum_ner_epochs:
            bad+=1
            if bad>=5:break
        # Do not count zero-F1 warm-up epochs against the five-epoch patience.
        if task=='ner' and epoch==minimum_ner_epochs:bad=0
    model.load_state_dict(beststate)
    test=infer_dataset(model,loaders['test'],task,dev)
    torch.save({'state_dict':model.state_dict(),'task':task,'tags':tags,'vocab_size':len(vocab),
                'hyperparams':{'freeze':freeze,'bidirectional':bidir,'dropout':dropout,
                               'random_init':random_init,'crf':use_crf},'best_epoch':last_epoch},MODELS/f'{name}.pt')
    plot_curves(hist,['train_loss','val_loss'],f'{name}_loss.png',f'{name}: learning curves')
    plot_curves(hist,['val_f1'],f'{name}_validation_f1.png',f'{name}: validation F1')
    result={'name':name,'task':task,'best_epoch':last_epoch,'epochs_run':len(hist),
            'best_validation_f1':best,'test_f1':test['primary_f1'],
            'test_metrics':test['metrics'],'training_seconds_per_epoch':round(float(np.mean([x['epoch_seconds'] for x in hist])),2),
            'history':hist}
    save_json(f'{name}_metrics.json',result)
    return model,loaders,test,result


def evaluate_pos(test):
    gs=[x for seq in test['gold'] for x in seq];ps=[x for seq in test['pred'] for x in seq]
    cm=confusion_matrix(gs,ps,labels=POS_TAGS)
    plot_confusion(cm,POS_TAGS,'pos_confusion.png','POS tagging: 12-tag confusion matrix')
    most=[]
    for i,g in enumerate(POS_TAGS):
        for j,p in enumerate(POS_TAGS):
            if i!=j and cm[i,j]>0:
                examples=[]
                for toks,gseq,pseq in zip(test['tokens'],test['gold'],test['pred']):
                    wrong=[k for k,(a,b) in enumerate(zip(gseq,pseq)) if a==g and b==p]
                    if wrong and len(examples)<2:
                        examples.append({'sentence':' '.join(toks),'mismatched_token':toks[wrong[0]],'gold':g,'predicted':p})
                most.append({'gold':g,'predicted':p,'confusions':int(cm[i,j]),'examples':examples})
    most=sorted(most,key=lambda x:-x['confusions'])[:3]
    save_json('pos_confused_tag_pairs.json',most)
    print('Top 3 POS confusions:',most)
    return most


def evaluate_ner(test):
    scores,fp,fn=conlleval_scores(test['gold'],test['pred'])
    def describe(errors):
        result=[]
        for e in errors[:5]:
            i=e['sentence_index'];a,b=e['span'];tokens=test['tokens'][i]
            gold_seq=test['gold'][i];pred_seq=test['pred'][i]
            gold_here=gold_seq[a:b];pred_here=pred_seq[a:b]
            expected=set(x.replace('B-','').replace('I-','') for x in gold_here if x!='O')
            predicted=set(x.replace('B-','').replace('I-','') for x in pred_here if x!='O')
            if e['error']=='false_positive':
                if expected and expected!=predicted:
                    why=f'Wrong entity type: gold spans show {sorted(expected)}, prediction shows {sorted(predicted)}.'
                elif expected:
                    why='Entity type overlaps, but predicted start/end boundary does not match the gold span.'
                else:
                    why='Model predicted an entity where the silver-label gazetteer marked no entity.'
            else:
                if predicted and expected!=predicted:
                    why=f'Model used the wrong entity type: expected {sorted(expected)}, predicted {sorted(predicted)}.'
                elif predicted:
                    why='Model predicted an entity but did not recover the complete reference span.'
                else:
                    why='Model missed a reference entity entirely, potentially due to lexical/contextual sparsity.'
            result.append({**e,'entity':' '.join(tokens[a:b]),'sentence':' '.join(tokens),
                           'gold_tags':gold_here,'predicted_tags':pred_here,'explanation':why})
        return result
    errors={'false_positives':describe(fp),'false_negatives':describe(fn),
            'error_warning':'Reference tags are generated automatically, not human-annotated gold.'}
    save_json('ner_error_analysis.json',errors)
    # conlleval compatible input file: token gold pred, one blank line per sentence.
    with (RES/'ner_conlleval_input.conll').open('w',encoding='utf-8') as f:
        for tok,gold,pred in zip(test['tokens'],test['gold'],test['pred']):
            for x,y,z in zip(tok,gold,pred):f.write(f'{x} {y} {z}\n')
            f.write('\n')
    # Re-read the actual CoNLL triplets through our independent conlleval-style CLI parser.
    from .conlleval import evaluate
    cli_scores=evaluate(str(RES/'ner_conlleval_input.conll'))
    assert abs(cli_scores['overall']['f1']-scores['overall']['f1'])<1e-10
    save_json('ner_conlleval_span_scores.json',cli_scores)
    with (RES/'ner_conlleval_metrics.txt').open('w',encoding='utf-8') as f:
        for typ,metrics in cli_scores.items():
            f.write(f'{typ} P={metrics["precision"]:.5f} R={metrics["recall"]:.5f} '
                    f'F1={metrics["f1"]:.5f} TP={metrics["TP"]} '
                    f'FP={metrics["FP"]} FN={metrics["FN"]}\n')
    print('NER entity-level CoNLL-style metrics:',cli_scores)
    return errors


def run():
    if not (DATA/'pos_train.conll').exists():
        from .annotation import run as annotate
        annotate()
    vocab=json.loads((EMB/'word2idx.json').read_text('utf-8'))
    weights=np.load(EMB/'embeddings_w2v.npy')
    if len(vocab)!=weights.shape[0]:raise ValueError('C3 embedding/vocabulary mismatch')
    summary={}
    for task in ['pos','ner']:
        modes={};mode_tests={}
        for mode in ['frozen','finetuned']:
            frozen=(mode=='frozen')
            model,loaders,test,result=train_one(task,f'bilstm_{task}_{mode}',vocab,weights,
                                                 freeze=frozen,use_crf=(task=='ner'))
            modes[mode]=result;mode_tests[mode]=test
        bestmode=max(('finetuned','frozen'),key=lambda k:modes[k]['best_validation_f1'])
        best_test=mode_tests[bestmode]
        # Copy EXACTLY the checkpoint selected by validation F1; this avoids an
        # unnecessary, potentially non-identical third training run.
        import shutil
        shutil.copyfile(MODELS/f'bilstm_{task}_{bestmode}.pt',MODELS/f'bilstm_{task}.pt')
        best_history=modes[bestmode]['history']
        plot_curves(best_history,['train_loss','val_loss'],f'bilstm_{task}_loss.png',
                    f'bilstm_{task}: selected {bestmode} learning curves')
        if task=='pos':evaluate_pos(best_test)
        else:evaluate_ner(best_test)
        summary[task]={'modes':{k:{'val_f1':v['best_validation_f1'],'test_f1':v['test_f1']} for k,v in modes.items()},
                       'chosen_mode':bestmode,'test_metrics':best_test['metrics'],
                       'best_epoch':modes[bestmode]['best_epoch']}
    # Each ablation trained independently with identical data split, random seed and hyperparams.
    opts={
       'A1_unidirectional':dict(bidir=False,use_crf=True),
       'A2_no_dropout':dict(dropout=0.,use_crf=True),
       'A3_random_embeddings':dict(random_init=True,use_crf=True),
       'A4_softmax_no_crf':dict(use_crf=False),
    }
    ablation={}
    for name,cfg in opts.items():
        _,_,res,info=train_one('ner',name,vocab,weights,freeze=False,**cfg)
        ablation[name]={'validation_F1':info['best_validation_f1'],'test_F1':res['primary_f1'],
                        'description':{'A1_unidirectional':'No backward context',
                         'A2_no_dropout':'Dropout removed',
                         'A3_random_embeddings':'Replace C3 embeddings with random initialization',
                         'A4_softmax_no_crf':'Token softmax replaces CRF/Viterbi'}[name]}
    summary['ablations']=ablation
    summary['ner_crf_vs_softmax']={'crf':summary['ner']['test_metrics']['overall']['f1'],
                                   'softmax':ablation['A4_softmax_no_crf']['test_F1']}
    summary['ner_initialization_comparison']={
        'C3_word2vec_main_F1':summary['ner']['test_metrics']['overall']['f1'],
        'random_embeddings_A3_F1':ablation['A3_random_embeddings']['test_F1'],
        'label_reference':'Automated silver gazetteer labels, NOT human-reviewed gold',
        'changes':'Pretrained C3 global rescaling, minimum NER warm-up, TRAIN-only entity-weighted CE'
    }
    save_json('part2_summary.json',summary)
    return summary

if __name__=='__main__':run()
