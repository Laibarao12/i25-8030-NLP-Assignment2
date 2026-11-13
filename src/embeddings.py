"""Part 1: exact TF-IDF, symmetric PPMI, PyTorch Skip-gram/negative sampling."""
from __future__ import annotations
from collections import Counter
import json, itertools, math, time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm as sparse_norm
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import (DATA, EMB, FIGS, RES, TOPICS, tokenize, load_text, load_articles,
                     vocab_from_docs, seed_all, save_json, device, split_numbered_articles, normalize)

WINDOW=5; K=10; BATCH=1024; EPOCHS=5
QUERIES=['Pakistan','Hukumat','Adalat','Maeeshat','Fauj','Sehat','Taleem','Aabadi']
SEED_SEMANTICS={
    'Politics':'pakistan hukumat adalat wazeer wazir intikhab assembly election government parliament سیاست حکومت عدالت وزیر انتخاب پاکستان'.split(),
    'Sports':'cricket match team player football captain score wicket کرکٹ میچ ٹیم کھلاڑی وکٹ کھیل'.split(),
    'Economy':'maeeshat bank business tijarat economy trade inflation budget budget معیشت تجارت بینک بجٹ مہنگائی'.split(),
    'International':'afghanistan india china iran america world united nations foreign افغانستان بھارت چین ایران امریکہ عالمی'.split(),
    'Health & Society':'sehat taleem school hospital doctor vaccine health education hospital صحت تعلیم اسکول ہسپتال ڈاکٹر'.split(),
}


def docs_from_text(name):
    text=load_text(name)
    numbered=split_numbered_articles(text)
    chunks=[body for _,body in numbered] if numbered is not None else [s.strip() for s in text.splitlines() if s.strip()]
    return [tok for chunk in chunks if (tok:=tokenize(chunk))]


def save_dense_npy_from_csr(mat,path):
    arr=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=mat.shape)
    arr[:]=0
    for start in range(0,mat.shape[0],256):
        end=min(start+256,mat.shape[0]);arr[start:end]=mat[start:end].toarray().astype('float32')
    arr.flush();del arr


def tfidf(articles,vocab):
    n=len(articles); rows=[];cols=[];vals=[]
    for i,article in enumerate(articles):
        c=Counter(vocab.get(t,1) for t in article['tokens'])
        for w,count in c.items():rows.append(i);cols.append(w);vals.append(count)
    x=sparse.csr_matrix((np.asarray(vals,dtype=np.float32),(rows,cols)),shape=(n,len(vocab)))
    df=np.asarray((x>0).sum(axis=0)).ravel()
    idf=np.log(n/(1.+df))  # exactly the assignment formula, not sklearn's smoothed version
    weighted=x.multiply(idf).tocsr()
    save_dense_npy_from_csr(weighted,EMB/'tfidf_matrix.npy')
    y=[a['topic'] for a in articles]
    inverse={v:k for k,v in vocab.items()}
    result={}
    for topic in TOPICS:
        sel=np.where(np.asarray(y)==topic)[0]
        if not len(sel):result[topic]=[];continue
        mean=np.asarray(weighted[sel].mean(axis=0)).ravel()
        rest=np.where(np.asarray(y)!=topic)[0]
        others=np.asarray(weighted[rest].mean(axis=0)).ravel() if len(rest) else np.zeros_like(mean)
        score=mean-others;score[:2]=-np.inf
        top=np.argsort(score)[::-1][:10]
        result[topic]=[{'word':inverse[int(i)],'topic_mean_tfidf':round(float(mean[i]),5),
                        'topic_minus_others':round(float(score[i]),5)} for i in top]
        print(topic, 'top 10:', [inverse[int(i)] for i in top])
    save_json('tfidf_top_words.json',result)
    print('Saved TF-IDF', weighted.shape, 'nonzeros:',weighted.nnz)
    return result


def ppmi_matrix(docs,vocab):
    """Directed symmetric +/- k counts in sparse CSR, saved as required dense .npy.
    Streaming COO->CSR chunks avoid holding all corpus pair arrays simultaneously.
    """
    v=len(vocab); acc=sparse.csr_matrix((v,v),dtype=np.float64)
    ii=[];jj=[]; chunk_target=1_500_000
    def flush():
        nonlocal acc,ii,jj
        if not ii:return
        batch=sparse.coo_matrix((np.ones(len(ii),dtype=np.float64),(ii,jj)),shape=(v,v)).tocsr()
        acc=acc+batch;ii.clear();jj.clear()
    for d in docs:
        ids=[vocab.get(t,1) for t in d]
        for ix,center in enumerate(ids):
            for j in range(max(0,ix-WINDOW),min(len(ids),ix+WINDOW+1)):
                if j!=ix:ii.append(center);jj.append(ids[j])
            if len(ii)>chunk_target:flush()
    flush()
    acc.eliminate_zeros()
    total=acc.sum()
    row=np.asarray(acc.sum(axis=1)).ravel(); col=np.asarray(acc.sum(axis=0)).ravel()
    coo=acc.tocoo()
    # log2(C_ij * total / (row_i * col_j)) for nonzero counts.
    ratio=np.asarray(coo.data*total/(row[coo.row]*col[coo.col]+1.e-30),dtype=np.float64)
    values=np.maximum(np.log2(np.maximum(ratio,1e-30)),0.)
    ppmi=sparse.csr_matrix((values.astype(np.float32),(coo.row,coo.col)),shape=(v,v))
    ppmi.eliminate_zeros()
    save_dense_npy_from_csr(ppmi,EMB/'ppmi_matrix.npy')
    print('Saved PPMI',ppmi.shape,'nnz',ppmi.nnz)
    return ppmi


# Assignment questions name queries in Roman Urdu but the A1 preprocessing
# deliberately removes Latin script. Translate for lookup, not for training.
ROMAN_QUERY_ALIASES={
    'pakistan':'پاکستان','hukumat':'حکومت','adalat':'عدالت',
    'maeeshat':'معیشت','fauj':'فوج','sehat':'صحت',
    'taleem':'تعلیم','aabadi':'آبادی',
    'islamabad':'اسلام آباد','karachi':'کراچی',
}


def resolve_word(word,vocab):
    """Find the corpus *surface or A1-stemmed* representation of one phrase.

    No word is inserted into the vocabulary for evaluation. Returns None
    if the word was not actually observed.
    """
    from .a1_linguistics import clean_and_process_article, urdu_stem, urdu_lemmatize
    raw=str(word).strip()
    translated=ROMAN_QUERY_ALIASES.get(raw.lower(),raw)
    candidates=[normalize(raw), normalize(translated)]
    for cand in (translated,):
        try:
            processed=clean_and_process_article(cand)
            if len(processed)==1 and len(processed[0].split())==1:
                candidates.append(normalize(processed[0]))
            candidates += [normalize(urdu_stem(cand)), normalize(urdu_lemmatize(cand))]
        except Exception:
            pass
    # Multi-word place names may be separate tokens; corpus is word-level.
    # Prefer a whole phrase only if the tokenizer did produce that exact token.
    for c in candidates:
        if c in vocab:return c
    return None



def resolve_indices(word,vocab):
    """Return in-vocab token ids, averaging word vectors for multiword names."""
    simple=resolve_word(word,vocab)
    if simple is not None:return [vocab[simple]]
    from .a1_linguistics import clean_and_process_article
    translated=ROMAN_QUERY_ALIASES.get(str(word).lower().strip(),str(word))
    clean=' '.join(clean_and_process_article(translated))
    parts=tokenize(clean)
    if len(parts)>=2 and all(t in vocab for t in parts):
        return [vocab[t] for t in parts]
    return None


def phrase_vector(emb, vocab, word):
    ids=resolve_indices(word,vocab)
    if ids is None:return None,None
    if sparse.issparse(emb):
        return np.asarray(emb[ids].mean(axis=0)).ravel(),ids
    return np.asarray(emb[ids],dtype=np.float32).mean(axis=0),ids

def cosine_neighbors(emb,vocab,word,n=10):
    vec,ids=phrase_vector(emb,vocab,word)
    if vec is None:return []
    norms=np.sqrt(np.asarray(emb.power(2).sum(axis=1)).ravel())+1e-12 if sparse.issparse(emb) else np.linalg.norm(emb,axis=1)+1e-9
    scores=np.asarray(emb@vec).ravel()/(norms*(np.linalg.norm(vec)+1e-9))
    for idx in ids:scores[idx]=-np.inf
    scores[:2]=-np.inf
    inv={v:k for k,v in vocab.items()}
    top=np.argsort(scores)[::-1][:n]
    return [(inv.get(int(i),'?'),round(float(scores[i]),4)) for i in top if np.isfinite(scores[i])]


def plot_tsne(ppmi,vocab,counts):
    inv={i:t for t,i in vocab.items()}
    top=[vocab[t] for t,_ in counts.most_common(200) if t in vocab and vocab[t]>1][:200]
    if len(top)<10:print('Too few tokens for t-SNE');return
    subset=ppmi[top].toarray()
    # SVD first provides an efficient, noise-reduced embedding for t-SNE.
    svd=TruncatedSVD(n_components=min(50,subset.shape[0]-1,subset.shape[1]-1),random_state=42)
    dense=svd.fit_transform(subset)
    points=TSNE(n_components=2,perplexity=min(30,(len(top)-1)//3),random_state=42,init='pca',learning_rate='auto').fit_transform(dense)
    name_map={w.lower():c for c,terms in SEED_SEMANTICS.items() for w in terms}
    labels=[name_map.get(inv[i].lower(),'Other / unlabelled') for i in top]
    palette=['#b54c5e','#256e96','#d28e3b','#6a52a4','#36917a','#878787']
    fig,ax=plt.subplots(figsize=(13,9))
    for cl,color in zip(TOPICS+['Other / unlabelled'],palette):
        mask=np.array([x==cl for x in labels]);
        if mask.any():ax.scatter(points[mask,0],points[mask,1],s=34,alpha=.76,label=cl,c=color)
    for i,wid in enumerate(top):ax.annotate(inv[wid],points[i],fontsize=6,alpha=.8)
    ax.set(xlabel='t-SNE dimension 1',ylabel='t-SNE dimension 2',title='PPMI: 200 frequent tokens (manual semantic seed labels)')
    ax.legend(loc='best',fontsize=9);fig.tight_layout();fig.savefig(FIGS/'ppmi_tsne.png',dpi=200);plt.close(fig)
    print('Saved semantic t-SNE plot; unlabeled tokens are not assumed to belong to a topic')


class SkipGram(nn.Module):
    def __init__(self,n,d):
        super().__init__();self.V=nn.Embedding(n,d,padding_idx=0);self.U=nn.Embedding(n,d,padding_idx=0)
        nn.init.uniform_(self.V.weight,-.5/d,.5/d);nn.init.zeros_(self.U.weight)
        with torch.no_grad():self.V.weight[0].zero_()
    def forward(self,centers,contexts,negatives):
        v=self.V(centers)
        pos=(v*self.U(contexts)).sum(dim=1)
        neg=torch.bmm(self.U(negatives),v.unsqueeze(2)).squeeze(2)
        return -(F.logsigmoid(pos)+F.logsigmoid(-neg).sum(dim=1)).mean()
    def mean_embeddings(self):
        return ((self.V.weight.detach()+self.U.weight.detach())*.5).cpu().numpy().astype(np.float32)


def pairs(docs,vocab,seed):
    """On-the-fly positive pairs; shuffle docs each epoch, fixed symmetric k=5."""
    rng=np.random.default_rng(seed)
    order=rng.permutation(len(docs))
    for di in order:
        ids=[vocab.get(t,1) for t in docs[di]]
        for ix in rng.permutation(len(ids)):
            w=ids[ix]
            if w==0:continue
            for j in range(max(0,ix-WINDOW),min(len(ids),ix+WINDOW+1)):
                if j!=ix and ids[j]!=0:yield (w,ids[j])


def train_sg(docs,vocab,counts,d,filename,epochs=EPOCHS, max_batches=None):
    dev=device();seed_all(42)
    print(f'Train {filename} | articles={len(docs)} | vocabulary={len(vocab)} | dim={d} | device={dev}')
    freqs=np.zeros(len(vocab),dtype=np.float64)
    for w,n in counts.items():freqs[vocab.get(w,1)]+=n
    freqs[:1]=0
    prob=np.power(freqs,.75);prob/=prob.sum()
    assert np.isclose(prob.sum(),1)
    prob_t=torch.tensor(prob,dtype=torch.float32,device=dev)
    model=SkipGram(len(vocab),d).to(dev);opt=torch.optim.Adam(model.parameters(),lr=.001)
    history=[];step=0
    for epoch in range(1,epochs+1):
        start=time.time();total_loss=0.;num_batches=0;buf=[]
        def update(buf):
            nonlocal step,total_loss,num_batches
            center=torch.tensor([a for a,b in buf],device=dev,dtype=torch.long)
            context=torch.tensor([b for a,b in buf],device=dev,dtype=torch.long)
            neg=torch.multinomial(prob_t,num_samples=len(buf)*K,replacement=True).reshape(-1,K)
            opt.zero_grad(set_to_none=True)
            loss=model(center,context,neg);loss.backward();opt.step()
            total_loss+=float(loss.item());num_batches+=1;step+=1
            if step%100==0:print(f'  epoch={epoch}, step={step}, batch loss={loss.item():.4f}')
        for pair in pairs(docs,vocab,seed=42+epoch):
            buf.append(pair)
            if len(buf)>=BATCH:
                update(buf);buf=[]
                if max_batches is not None and num_batches>=max_batches:break
        if buf and (max_batches is None or num_batches<max_batches):update(buf)
        if not num_batches:raise ValueError('No skip-gram training pairs available')
        mean=total_loss/num_batches
        history.append({'epoch':epoch,'train_loss':mean,'batches':num_batches,'seconds':round(time.time()-start,2)})
        print(f'epoch {epoch}/{epochs} average loss {mean:.4f} over {num_batches} batches')
    np.save(EMB/filename,model.mean_embeddings())
    fig,ax=plt.subplots(figsize=(7,4));ax.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],marker='o')
    ax.set(xlabel='Epoch',ylabel='Mean skip-gram negative sampling loss',title=f'Skip-gram training loss: {filename}')
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(FIGS/f'{Path(filename).stem}_loss.png',dpi=170);plt.close(fig)
    save_json(f'{Path(filename).stem}_training.json',history)
    return model.mean_embeddings(),history


def analogy_results(emb,vocab):
    tests=json.loads((Path(__file__).resolve().parents[1]/'assets'/'analogies.json').read_text('utf-8'))
    arr=emb.toarray() if sparse.issparse(emb) else np.asarray(emb)
    norms=np.linalg.norm(arr,axis=1)+1e-9
    inv={v:k for k,v in vocab.items()};rows=[]
    for r in tests:
        a,ai=phrase_vector(arr,vocab,r['a']);b,bi=phrase_vector(arr,vocab,r['b']);c,ci=phrase_vector(arr,vocab,r['c'])
        gold=resolve_word(r['gold'],vocab)
        if any(x is None for x in (a,b,c)):
            rows.append({**r,'candidates':[],'correct_top3':None,'status':'OOV premise'});continue
        target=b-a+c
        score=arr@target/(norms*(np.linalg.norm(target)+1e-9))
        for idx in set(ai+bi+ci):score[idx]=-np.inf
        score[:2]=-np.inf
        rank=np.argsort(score)[::-1][:3]
        candidates=[inv[int(i)] for i in rank]
        rows.append({**r,'candidates':candidates,'correct_top3':(gold in candidates) if gold is not None else None,
                     'status':'evaluated' if gold is not None else 'gold OOV'})
    return rows


def mrr_20(emb,vocab):
    test=json.loads((Path(__file__).resolve().parents[1]/'assets'/'mrr_20_pairs.json').read_text('utf-8'))
    if len(test)!=20:raise ValueError('MRR gold file must have 20 human-labelled word pairs')
    arr=emb.toarray() if sparse.issparse(emb) else np.asarray(emb)
    norms=np.linalg.norm(arr,axis=1)+1e-9
    scored=[]
    for r in test:
        query,exclude=phrase_vector(arr,vocab,r['query']); gold=resolve_word(r['gold'],vocab)
        if query is None or gold is None:
            scored.append({**r,'reciprocal_rank':None,'status':'OOV'});continue
        score=arr@query/(norms*(np.linalg.norm(query)+1e-9))
        for idx in exclude:score[idx]=-np.inf
        score[:2]=-np.inf
        rank=int(np.sum(score>score[vocab[gold]])+1)
        scored.append({**r,'rank':rank,'reciprocal_rank':1./rank,'status':'evaluated'})
    valid=[x['reciprocal_rank'] for x in scored if x['reciprocal_rank'] is not None]
    return {'MRR_evaluable':float(np.mean(valid)) if valid else None,
            'evaluable_pairs':len(valid),'requested_pairs':len(test),'detail':scored}


def evaluate_conditions(mat,vocs):
    query_words=QUERIES+['islamabad','karachi']
    out={}
    for cond,emb in mat.items():
        vocab=vocs[cond]
        neigh={q:cosine_neighbors(emb,vocab,q,5) for q in query_words[:5]}
        metric=mrr_20(emb,vocab)
        out[cond]={'neighbors_five_queries':neigh,'MRR':metric['MRR_evaluable'],
                   'evaluable_pairs':metric['evaluable_pairs'],'total_pairs':20}
        print(cond,'MRR:',metric['MRR_evaluable'],'eligible:',metric['evaluable_pairs'])
        for q,v in neigh.items():print(' ',q,v)
    save_json('four_condition_comparison.json',out)
    return out


def run():
    seed_all(42)
    articles=load_articles(require_labels=False)
    cleaned=[a['tokens'] for a in articles]
    raw=docs_from_text('raw.txt')
    vocab,counts=vocab_from_docs(cleaned,10000)
    (EMB/'word2idx.json').write_text(json.dumps(vocab,ensure_ascii=False,indent=2),'utf-8')
    topwords=tfidf(articles,vocab)
    ppmi=ppmi_matrix(cleaned,vocab)
    plot_tsne(ppmi,vocab,counts)
    p_neighbors={q:cosine_neighbors(ppmi,vocab,q,5) for q in QUERIES+['islamabad','karachi']}
    save_json('ppmi_neighbors.json',p_neighbors)
    print('\nPPMI neighbors:');[print(k,v) for k,v in p_neighbors.items()]
    rawv,rawcounts=vocab_from_docs(raw,10000)
    (EMB/'raw_word2idx.json').write_text(json.dumps(rawv,ensure_ascii=False,indent=2),'utf-8')
    c2,_=train_sg(raw,rawv,rawcounts,100,'embeddings_w2v_raw.npy')
    c3,_=train_sg(cleaned,vocab,counts,100,'embeddings_w2v.npy')
    c4,_=train_sg(cleaned,vocab,counts,200,'embeddings_w2v_200.npy')
    queries={q:cosine_neighbors(c3,vocab,q,10) for q in QUERIES}
    save_json('w2v_neighbors.json',queries)
    print('Required eight query words:');[print(k,v) for k,v in queries.items()]
    analogies=analogy_results(c3,vocab);save_json('analogies.json',analogies)
    print('Analogies (top3 correct / evaluable):',sum(x['correct_top3'] is True for x in analogies),'/',sum(x['correct_top3'] is not None for x in analogies))
    comparison=evaluate_conditions({'C1_PPMI':ppmi,'C2_raw':c2,'C3_cleaned':c3,'C4_d200':c4},
                                   {'C1_PPMI':vocab,'C2_raw':rawv,'C3_cleaned':vocab,'C4_d200':vocab})
    return comparison

if __name__=='__main__':run()