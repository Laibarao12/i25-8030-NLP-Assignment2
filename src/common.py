"""Corpus parsing, repeatable splits, plots, and results logging for DS5007 A2.
No pretrained weights, Gensim, HuggingFace or built-in Transformer modules.
"""
from __future__ import annotations
import json, re, random, unicodedata, os
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
EMB = ROOT / 'embeddings'
MODELS = ROOT / 'models'
FIGS = ROOT / 'figures'
RES = ROOT / 'results'
for p in [DATA, EMB, MODELS, FIGS, RES]: p.mkdir(parents=True, exist_ok=True)
TOPICS = ['Politics', 'Sports', 'Economy', 'International', 'Health & Society']
STOP = {'<pad>':0,'<unk>':1}
URDU_DIACRITICS = re.compile('[\u064b-\u065f\u0670\u06d6-\u06ed]')
TOK_RE = re.compile(r"<num>|[\u0621-\u06d3\u06fa-\u06ff]+|[A-Za-z]+(?:['’][A-Za-z]+)?|\d+(?:[.,]\d+)*|[^\w\s]", re.UNICODE)
SENT_RE = re.compile(r'(?<=[.!?؟۔])\s+|\n+')


def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def normalize(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    s = URDU_DIACRITICS.sub('', s)
    return s.replace('ي','ی').replace('ك','ک').replace('\u0640','').strip()


def tokenize(s, keep_punc=False):
    ts = TOK_RE.findall(normalize(s))
    return ts if keep_punc else [t for t in ts if any(ch.isalnum() for ch in t)]


def load_text(name):
    path = DATA/name
    if not path.exists():
        raise FileNotFoundError(f'Missing required file: {path}. Copy your actual Assignment 1 corpus here.')
    return path.read_text('utf-8-sig', errors='replace')


def load_metadata():
    path=DATA/'Metadata.json'
    if not path.exists(): raise FileNotFoundError(f'Missing required file: {path}')
    x=json.loads(path.read_text('utf-8-sig'))
    if isinstance(x,list): return x
    if isinstance(x,dict):
        for key in ['articles','documents','records','data','items','news']:
            if key in x and isinstance(x[key],list): return x[key]
        if all(isinstance(v,dict) for v in x.values()): return list(x.values())
    raise ValueError('Unexpected Metadata.json structure; revise load_metadata() to match the real source.')


def label_topic(rec):
    """Choose five-class labels only from unambiguous metadata or explicit title keywords.

    BBC's scraped category field frequently contains a *site-wide navigation breadcrumb*.
    Those breadcrumbs are not category annotations and must not be trusted.
    """
    if isinstance(rec,str): rec={'category':rec}
    patterns={
      'Politics':['politic','government','election','parliament','siyasat','hukumat','assembly','سیاس','حکومت','انتخاب'],
      'Sports':['sport','cricket','football','khel','match','کھیل','کرکٹ'],
      'Economy':['econom','business','finance','maeeshat','budget','trade','معیش','کاروبار','اقتصاد'],
      'International':['international','world','foreign','dunya','عالمی','بین الاقوامی','خارجہ','دنیا'],
      'Health & Society':['health','society','education','sehat','taleem','social','science','صحت','تعلیم','معاشر'],
    }
    # Prioritize clearly labeled topic/section fields; don't assume page navigation is a label.
    for key in ('topic','label','class','news_category','genre','section','category'):
        value=str(rec.get(key,'')).strip()
        if not value:continue
        low=value.lower()
        # Original Assignment 1's BBC breadcrumb is typically an entire navigation bar.
        if ('صفحۂ اول' in value or 'ویڈیو سیکشن' in value or
            len(value.split())>9 or ',' in value):continue
        for label in TOPICS:
            if low==label.lower():return label,'metadata'
        hits=[topic for topic,patterns_for_topic in patterns.items()
              if any(word in low for word in patterns_for_topic)]
        if len(hits)==1:return hits[0],'metadata'
    # Title is only weak supervision: a human must validate the review file before grading.
    txt=' '.join(str(rec.get(k,'')) for k in ('title','headline')).lower()
    kw={
       'Politics':['election','government','minister','parliament','imran','hukumat','انتخاب','حکومت','وزیر','پارلیمان'],
       'Sports':['cricket','match','team','player','score','khel','کرکٹ','میچ','کھلاڑی','ٹیم'],
       'Economy':['inflation','trade','bank','gdp','budget','maeeshat','مہنگائی','تجارت','بینک','بجٹ'],
       'International':['united nations','treaty','foreign','bilateral','conflict','اقوام متحدہ','معاہدہ','خارجہ','سرحد'],
       'Health & Society':['hospital','disease','vaccine','flood','education','ہسپتال','بیماری','ویکسین','سیلاب','تعلیم'],
    }
    counts={k:sum(txt.count(w) for w in words) for k,words in kw.items()}
    if max(counts.values())>0:
        top=max(counts.values());winners=[k for k,n in counts.items() if n==top]
        if len(winners)==1:return winners[0],'title_keyword_weak_label'
    return None,'unlabelled'


def split_numbered_articles(text):
    """Return [(article_id, body)] from the exact A1 '### ARTICLE N ###' structure.

    Return None for unrelated formats. Don't merge/split on blank lines inside articles.
    """
    pattern=r'(?m)^### ARTICLE\s+(\d+)\s+###\s*$'
    m=list(re.finditer(pattern,text))
    if not m:return None
    if text[:m[0].start()].strip():
        raise ValueError('Unexpected content before first A1 article marker.')
    out=[];seen=set()
    for i,match in enumerate(m):
        number=int(match.group(1))
        if number in seen:raise ValueError(f'Duplicate A1 article marker: {number}')
        seen.add(number)
        body=text[match.end():m[i+1].start() if i+1<len(m) else len(text)].strip()
        if not body:raise ValueError(f'A1 article {number} is empty')
        out.append((number,body))
    return out


def load_articles(require_labels=True):
    """Align cleaned A1 article markers with Metadata.json IDs (not arbitrary row order).

    Optional data/topic_labels_review.csv manual topic overrides take precedence. This
    file must contain columns article_number and topic; no fake topics are created.
    """
    import csv
    text=load_text('cleaned.txt'); records=load_metadata()
    numbered=split_numbered_articles(text)
    by_id={}
    if numbered is not None:
        if len({k for k,_ in numbered})!=len(numbered):raise ValueError('Duplicate article ID')
        by_id=dict(numbered)
        if not all(isinstance(r,dict) and 'article_number' in r for r in records):
            raise ValueError('A1 numbered corpus needs article_number in every Metadata.json record')
        metadata_ids=[int(r['article_number']) for r in records]
        if len(set(metadata_ids))!=len(metadata_ids):raise ValueError('Duplicate metadata article_number')
        if set(metadata_ids)!=set(by_id):
            raise ValueError(f'A1 article ID mismatch. Only in cleaned: {sorted(set(by_id)-set(metadata_ids))[:10]}; '
                             f'only in metadata: {sorted(set(metadata_ids)-set(by_id))[:10]}')
        sep='A1 numbered article blocks with verified IDs'
    else:
        by_line=[s.strip() for s in text.splitlines() if s.strip()]
        by_blank=[s.strip().replace('\n',' ') for s in re.split(r'\n\s*\n',text) if s.strip()]
        if len(by_line)==len(records):docs=by_line;sep='one line per article'
        elif len(by_blank)==len(records):docs=by_blank;sep='blank-line-separated articles'
        else:
            raise ValueError(f'Cannot align corpus with metadata: {len(by_line)} non-empty lines, '
                             f'{len(by_blank)} blank blocks, {len(records)} metadata records')
        records=[dict(r,article_number=r.get('article_number',i+1)) if isinstance(r,dict) else
                 {'article_number':i+1,'category':str(r)} for i,r in enumerate(records)]
        by_id={int(r['article_number']):doc for r,doc in zip(records,docs)}
    overrides={}
    path=DATA/'topic_labels_review.csv'
    if path.exists():
        with path.open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                aid=row.get('article_number','').strip()
                if not aid:continue
                topic=row.get('topic','').strip()
                if topic:
                    if topic not in TOPICS:raise ValueError(f'Invalid topic override {topic} for article {aid}')
                    overrides[int(aid)]=topic
        extra=set(overrides)-set(by_id)
        if extra:raise ValueError(f'Unknown article numbers in topic labels: {sorted(extra)}')
    out=[];sources=[]
    for r in records:
        article_id=int(r['article_number'])
        if article_id in overrides:label,source=overrides[article_id],'manual_review'
        else:label,source=label_topic(r)
        sources.append(source)
        if require_labels and label is None:continue
        s=by_id[article_id]
        out.append({'id':article_id,'text':s,'tokens':tokenize(s),'topic':label,'label_source':source})
    if require_labels and not out:raise ValueError('No labelled articles; review data/topic_labels_review.csv')
    print(f'Loaded {len(out)}/{len(records)} aligned articles ({sep}); '
          f'label sources: {dict(Counter(sources))}')
    return out


def vocab_from_docs(docs,max_tokens=10000):
    counts=Counter(t for doc in docs for t in doc)
    vocab=dict(STOP)
    for t,_ in counts.most_common(max_tokens):
        if t not in vocab:vocab[t]=len(vocab)
    return vocab,counts


def stratified_split(items,labels,seed=42):
    inds=np.arange(len(items)); counts=Counter(labels)
    if len(counts)<2 or min(counts.values())<7:
        raise ValueError(f'Not enough examples per category for stratified 70/15/15: {counts}')
    tr,temp=train_test_split(inds,test_size=.30,stratify=labels,random_state=seed)
    va,te=train_test_split(temp,test_size=.5,stratify=np.asarray(labels)[temp],random_state=seed)
    return [list(map(int,x)) for x in (tr,va,te)]


def save_json(name,obj):
    path=RES/name
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),'utf-8')
    return path


def show_counts(name, labels, indices=None):
    counts=Counter(labels[i] for i in indices) if indices is not None else Counter(labels)
    print(f'{name}:', dict(sorted(counts.items())))
    return dict(counts)


def plot_curves(rows, keys, name, title):
    fig,ax=plt.subplots(figsize=(8,4.5))
    for key in keys: ax.plot([r['epoch'] for r in rows], [r.get(key,np.nan) for r in rows],marker='o',markersize=3,label=key)
    ax.set(xlabel='Epoch',ylabel='Metric value',title=title)
    ax.legend();ax.grid(alpha=.2); fig.tight_layout();fig.savefig(FIGS/name,dpi=170);plt.close(fig)


def plot_confusion(cm,labels,name,title):
    fig,ax=plt.subplots(figsize=(8,6))
    im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
    ax.set_xticks(range(len(labels)), labels, rotation=45,ha='right')
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel='Predicted label',ylabel='True label',title=title)
    if len(labels)<=6:
        for i in range(len(labels)):
            for j in range(len(labels)):ax.text(j,i,int(cm[i,j]),ha='center',va='center',fontsize=8)
    fig.tight_layout();fig.savefig(FIGS/name,dpi=170);plt.close(fig)


def device():return torch.device('cuda' if torch.cuda.is_available() else 'cpu')