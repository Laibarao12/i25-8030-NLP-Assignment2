"""Part 3: handwritten four-layer pre-LN Transformer encoder + fair BiLSTM classifier."""
from __future__ import annotations
import math, copy, json, time
from collections import Counter
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .common import (DATA, EMB, MODELS, FIGS, RES, TOPICS,load_articles,stratified_split,
                     seed_all,device,plot_curves,plot_confusion,save_json,show_counts)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAX_TOKENS=256; D_MODEL=128; HEADS=4; D_K=32; D_FF=512; NUM_LAYERS=4


class ScaledDotProductAttention(nn.Module):
    def __init__(self):super().__init__()
    def forward(self,q,k,v,key_mask=None):
        logits=(q@k.transpose(-2,-1))/math.sqrt(q.shape[-1])
        if key_mask is not None:
            # key_mask True=real token; broadcast over all query positions.
            logits=logits.masked_fill(~key_mask[:,None,:],-1e9)
        weights=F.softmax(logits,dim=-1)
        out=weights@v
        return out,weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model=D_MODEL,h=HEADS,dk=D_K):
        super().__init__();assert h*dk==d_model
        # Distinct nn.Linear per head, not one combined QKV matrix.
        self.q=nn.ModuleList([nn.Linear(d_model,dk) for _ in range(h)])
        self.k=nn.ModuleList([nn.Linear(d_model,dk) for _ in range(h)])
        self.v=nn.ModuleList([nn.Linear(d_model,dk) for _ in range(h)])
        self.attention=ScaledDotProductAttention();self.out=nn.Linear(h*dk,d_model)
    def forward(self,x,key_mask=None):
        features=[];attentions=[]
        for wq,wk,wv in zip(self.q,self.k,self.v):
            feature,att=self.attention(wq(x),wk(x),wv(x),key_mask)
            features.append(feature);attentions.append(att)
        return self.out(torch.cat(features,dim=-1)),torch.stack(attentions,dim=1)


class FeedForward(nn.Module):
    def __init__(self,d_model=D_MODEL,d_ff=D_FF,dropout=.1):
        super().__init__();self.w1=nn.Linear(d_model,d_ff);self.w2=nn.Linear(d_ff,d_model)
        self.drop=nn.Dropout(dropout)
    def forward(self,x):return self.w2(self.drop(F.relu(self.w1(x))))


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self,d_model=D_MODEL,max_positions=MAX_TOKENS+1):
        super().__init__()
        p=torch.arange(max_positions,dtype=torch.float32).unsqueeze(1)
        div=torch.exp(torch.arange(0,d_model,2,dtype=torch.float32)*(-math.log(10000.)/d_model))
        encoding=torch.zeros(1,max_positions,d_model,dtype=torch.float32)
        encoding[0,:,0::2]=torch.sin(p*div)
        encoding[0,:,1::2]=torch.cos(p*div)
        self.register_buffer('pe',encoding,persistent=True)  # fixed, non-learned
    def forward(self,x):return x+self.pe[:,:x.size(1)]


class PreLNEncoderBlock(nn.Module):
    def __init__(self,d_model=D_MODEL,d_ff=D_FF,dropout=.1):
        super().__init__();self.ln1=nn.LayerNorm(d_model);self.ln2=nn.LayerNorm(d_model)
        self.mhsa=MultiHeadSelfAttention(d_model);self.ffn=FeedForward(d_model,d_ff,dropout)
        self.drop=nn.Dropout(dropout)
    def forward(self,x,mask=None):
        out,att=self.mhsa(self.ln1(x),mask)
        x=x+self.drop(out)
        x=x+self.drop(self.ffn(self.ln2(x)))
        return x,att


class TopicTransformer(nn.Module):
    def __init__(self,vocab_size,num_classes=5):
        super().__init__();self.embed=nn.Embedding(vocab_size,D_MODEL,padding_idx=0)
        self.cls_token=nn.Parameter(torch.empty(1,1,D_MODEL));nn.init.normal_(self.cls_token,std=.02)
        self.pe=SinusoidalPositionEncoding(D_MODEL)
        self.blocks=nn.ModuleList([PreLNEncoderBlock() for _ in range(NUM_LAYERS)])
        self.norm=nn.LayerNorm(D_MODEL)
        self.classifier=nn.Sequential(nn.Linear(D_MODEL,64),nn.ReLU(),nn.Linear(64,num_classes))
    def forward(self,tokens,mask,return_attention=False):
        B=tokens.shape[0]
        x=self.embed(tokens)*math.sqrt(D_MODEL)
        x=torch.cat([self.cls_token.expand(B,-1,-1),x],dim=1)
        m=torch.cat([torch.ones(B,1,dtype=torch.bool,device=mask.device),mask],dim=1)
        x=self.pe(x)
        attention=None
        for block in self.blocks:x,attention=block(x,m)
        logits=self.classifier(self.norm(x[:,0]))
        return (logits,attention) if return_attention else logits


class BiLSTMTopicClassifier(nn.Module):
    """A separate article classifier: sequence tagger F1 is not comparable to article accuracy."""
    def __init__(self,weights):
        super().__init__();self.embed=nn.Embedding.from_pretrained(torch.tensor(weights,dtype=torch.float32),
                                                                     freeze=False,padding_idx=0)
        self.lstm=nn.LSTM(weights.shape[1],64,batch_first=True,bidirectional=True,num_layers=2,dropout=.5)
        self.drop=nn.Dropout(.5);self.fc=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Linear(64,5))
    def forward(self,x,mask):
        length=mask.sum(1).clamp(min=1).cpu()
        packed=nn.utils.rnn.pack_padded_sequence(self.embed(x),length,batch_first=True,enforce_sorted=False)
        _,(h,_)=self.lstm(packed)
        out=torch.cat([h[-2],h[-1]],dim=-1)
        return self.fc(self.drop(out))


class ArticleDataset(Dataset):
    def __init__(self,articles,vocab,indices):
        self.items=[articles[i] for i in indices];self.vocab=vocab
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        s=self.items[i]
        words=s['tokens'][:MAX_TOKENS]
        ids=[self.vocab.get(w,1) for w in words]
        x=np.zeros(MAX_TOKENS,dtype=np.int64);x[:len(ids)]=ids
        return torch.tensor(x),torch.tensor(TOPICS.index(s['topic']),dtype=torch.long),words,s['id']


def batch_collate(batch):
    xs,ys,words,ident=zip(*batch)
    x=torch.stack(xs);y=torch.stack(ys)
    return x,y,x.ne(0),list(words),list(ident)


def data_loaders():
    articles=load_articles();vocab=json.loads((EMB/'word2idx.json').read_text('utf-8'))
    categories=[a['topic'] for a in articles]
    if set(categories)!=set(TOPICS):
        raise ValueError('All 5 topic classes must be present; found '+str(Counter(categories)))
    tr,va,te=stratified_split(articles,categories)
    split={s:show_counts(s,categories,inds) for s,inds in [('train',tr),('val',va),('test',te)]}
    save_json('transformer_category_distribution.json',split)
    loaders={name:DataLoader(ArticleDataset(articles,vocab,ids),batch_size=12,
                             shuffle=name=='train',collate_fn=batch_collate)
             for name,ids in [('train',tr),('val',va),('test',te)]}
    return articles,vocab,loaders


def predict(model,loader,dev,return_rows=False):
    model.eval();all_y=[];all_p=[];all_rows=[];loss_sum=0.;n=0
    with torch.no_grad():
        for x,y,m,tokens,idx in loader:
            x=x.to(dev);y=y.to(dev);m=m.to(dev)
            z=model(x,m);loss_sum+=float(F.cross_entropy(z,y).item());n+=1
            pred=z.argmax(1).cpu().tolist();all_p.extend(pred);all_y.extend(y.cpu().tolist())
            if return_rows:
                for i in range(len(idx)):
                    all_rows.append({'article_id':idx[i],'tokens':tokens[i],
                                     'gold':int(y[i]),'pred':int(pred[i])})
    met={'accuracy':float(accuracy_score(all_y,all_p)),
         'macro_f1':float(f1_score(all_y,all_p,average='macro',labels=list(range(5)),zero_division=0)),
         'loss':loss_sum/max(n,1)}
    return met,all_y,all_p,all_rows


def cosine_warmup_scheduler(optimizer,total_steps,warmup_steps=50):
    def fac(step):
        step+=1
        if step<=warmup_steps:return step/max(1,warmup_steps)
        progress=min(1.,(step-warmup_steps)/max(1,total_steps-warmup_steps))
        return .5*(1.+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=fac)


def train_classifier(model,loaders,name,transformer=False):
    seed_all(42);dev=device();model=model.to(dev)
    lr=.0005;optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.01)
    epochs=20;total_steps=epochs*len(loaders['train']);scheduler=cosine_warmup_scheduler(optimizer,total_steps,50)
    history=[];best=-1.;beststate=None;bestep=None
    for ep in range(1,epochs+1):
        model.train();t0=time.perf_counter();tl=0.;total=0;correct=0
        for x,y,m,_,_ in loaders['train']:
            x=x.to(dev);y=y.to(dev);m=m.to(dev)
            optimizer.zero_grad(set_to_none=True);z=model(x,m)
            loss=F.cross_entropy(z,y);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step()
            tl+=float(loss.item())*len(y);total+=len(y);correct+=int((z.argmax(1)==y).sum())
        if dev.type=='cuda':torch.cuda.synchronize()
        secs=time.perf_counter()-t0
        vm,_,_,_=predict(model,loaders['val'],dev)
        history.append({'epoch':ep,'train_loss':tl/total,'val_loss':vm['loss'],
                        'train_accuracy':correct/total,'val_accuracy':vm['accuracy'],
                        'val_macro_f1':vm['macro_f1'],'seconds_per_epoch':round(secs,3)})
        print(name,'epoch',ep,'train acc',round(correct/total,3),'val acc',round(vm['accuracy'],3))
        if vm['macro_f1']>best+1e-8:
            best=vm['macro_f1'];beststate=copy.deepcopy(model.state_dict());bestep=ep
    # Requirement: all 20 epochs trained. Report checkpoint selected by validation only.
    model.load_state_dict(beststate)
    met,ys,ps,rows=predict(model,loaders['test'],dev,return_rows=True)
    torch.save({'model_state_dict':model.state_dict(),'best_epoch':bestep,
                'architecture':name,'num_classes':5},MODELS/f'{name}.pt')
    plot_curves(history,['train_loss','val_loss'],f'{name}_loss.png',name+': loss across 20 epochs')
    plot_curves(history,['train_accuracy','val_accuracy'],f'{name}_accuracy.png',name+': accuracy across 20 epochs')
    cm=confusion_matrix(ys,ps,labels=list(range(5)))
    plot_confusion(cm,TOPICS,f'{name}_confusion.png',name+': test confusion matrix')
    out={'test':met,'best_epoch_by_validation_F1':bestep,'best_val_F1':best,
         'training_seconds_per_epoch':round(float(np.mean([r['seconds_per_epoch'] for r in history])),3),
         'history':history,'confusion_matrix':cm.tolist()}
    save_json(f'{name}_metrics.json',out)
    return model,out,rows


def attention_figures(model,rows,vocab):
    dev=device();model.eval()
    correct=[r for r in rows if r['pred']==r['gold']]
    if len(correct)<3:
        print('WARNING: fewer than 3 correct test articles. Do not fabricate attention heatmaps.')
    figures=[];observations=[]
    for pos,r in enumerate(correct[:3],1):
        words=r['tokens'][:31]
        ids=[vocab.get(w,1) for w in words]+[0]*(MAX_TOKENS-len(words))
        x=torch.tensor(ids,device=dev,dtype=torch.long)[None,:];mask=x.ne(0)
        with torch.no_grad():
            z,att=model(x,mask,return_attention=True)
        # Two heads of last encoder layer; show first 32 tokens including CLS.
        n=len(words)+1
        weights=att.detach().cpu().numpy()[0,:,:n,:n]
        names=['[CLS]']+words
        for head in [0,1]:
            # CLS is the classification aggregator; inspect its top attended non-CLS keys.
            strongest=np.argsort(weights[head,0,1:])[::-1][:3]+1
            observations.append({'article_id':r['article_id'],'head':head+1,
                                 'top_cls_attention_words':[names[int(j)] for j in strongest],
                                 'attention_probabilities':[float(weights[head,0,int(j)]) for j in strongest]})
            fig,ax=plt.subplots(figsize=(max(8,n*.35),max(7,n*.32)))
            im=ax.imshow(weights[head],cmap='viridis',aspect='auto',vmin=0)
            fig.colorbar(im,ax=ax,label='Attention probability')
            ax.set_xticks(range(n),names,rotation=90,fontsize=7)
            ax.set_yticks(range(n),names,fontsize=7)
            ax.set(xlabel='Key token',ylabel='Query token',
                   title=f'Test article {r["article_id"]}: final-layer attention head {head+1} (correct: {TOPICS[r["gold"]]})')
            fig.tight_layout();filename=f'attention_article{pos}_head{head+1}.png';fig.savefig(FIGS/filename,dpi=180);plt.close(fig)
            figures.append(filename)
    return figures,observations


def comparison_paragraphs(bi,tr,observations):
    a=bi['test']['accuracy'];b=tr['test']['accuracy'];winner='Transformer' if b>a else ('BiLSTM' if a>b else 'Neither model')
    diff=abs(b-a)*100
    faster='BiLSTM' if bi['training_seconds_per_epoch']<tr['training_seconds_per_epoch'] else 'Transformer'
    converge='BiLSTM' if bi['best_epoch_by_validation_F1']<tr['best_epoch_by_validation_F1'] else ('Transformer' if bi['best_epoch_by_validation_F1']>tr['best_epoch_by_validation_F1'] else 'Neither')
    # These descriptions are computed from the *actual* final-layer attention weights.
    if observations:
        focus0=', '.join(observations[0]['top_cls_attention_words'])
        focus1=', '.join(observations[1]['top_cls_attention_words']) if len(observations)>1 else 'not available'
        att_sentence=f'For the first correctly classified article, the final layer CLS query had high attention on {focus0} in head 1.'
        att_second=f'The corresponding second head highlighted {focus1}; inspecting their differences can reveal whether heads distribute attention differently.'
    else:
        att_sentence='No correct held-out predictions were available for qualitative attention interpretation.'
        att_second='Therefore no attention-specific word-importance claim is justified for this run.'
    # 12 sentences, all requested points, honest about what attention can and cannot prove.
    sentences=[
      f'Both models were evaluated on the same stratified 5-class BBC Urdu article split.',
      f'The BiLSTM achieved {a*100:.2f}% test accuracy, while the Transformer achieved {b*100:.2f}%.',
      f'{winner} had the higher accuracy, with an absolute difference of {diff:.2f} percentage points.',
      f'The BiLSTM reached its best validation macro-F1 at epoch {bi["best_epoch_by_validation_F1"]}.',
      f'The Transformer reached its best validation macro-F1 at epoch {tr["best_epoch_by_validation_F1"]}.',
      f'{converge} reached its selected validation checkpoint in fewer epochs.',
      f'The average BiLSTM epoch took {bi["training_seconds_per_epoch"]:.2f} seconds, compared with {tr["training_seconds_per_epoch"]:.2f} seconds for the Transformer.',
      f'{faster} was faster per epoch in the measured runtime, and the difference can reflect recurrent sequential processing versus attention-matrix computation and the different parameter counts.',
      att_sentence,
      att_second,
      'For a dataset of just 200 to 300 articles, a smaller BiLSTM may be the safer first model because it has a lower capacity and can overfit less easily.',
      'However, the preferred architecture should be selected from repeated stratified evaluations, not assumed based on architecture alone.'
    ]
    assert len(sentences)==12
    (RES/'bilstm_vs_transformer_12_sentences.txt').write_text(' '.join(sentences),'utf-8')
    return sentences


def run():
    seed_all(42)
    articles,vocab,loaders=data_loaders()
    if not (EMB/'embeddings_w2v.npy').exists():raise FileNotFoundError('Run Part 1 first (C3 embeddings)')
    bi,bi_results,bi_rows=train_classifier(BiLSTMTopicClassifier(np.load(EMB/'embeddings_w2v.npy')),
                                         loaders,'bilstm_topic')
    tr,tr_results,tr_rows=train_classifier(TopicTransformer(len(vocab)),loaders,'transformer_cls',transformer=True)
    figures,attention_observations=attention_figures(tr,tr_rows,vocab)
    comparison=comparison_paragraphs(bi_results,tr_results,attention_observations)
    summary={'bilstm':bi_results['test'],'transformer':tr_results['test'],
             'bilstm_selected_epoch':bi_results['best_epoch_by_validation_F1'],
             'transformer_selected_epoch':tr_results['best_epoch_by_validation_F1'],
             'attention_heatmaps':figures,'attention_observations':attention_observations,
             'comparison_sentences':comparison}
    save_json('part3_summary.json',summary)
    return summary

if __name__=='__main__':run()