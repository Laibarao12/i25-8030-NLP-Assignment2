"""Mathematical smoke tests only: never mistaken for BBC Urdu experiment results."""
import unittest, tempfile, itertools
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scipy import sparse
from src.common import stratified_split
from src.annotation import build_lexicon,gazetteer,ner_bio,POS_TAGS
from src.embeddings import ppmi_matrix,cosine_neighbors,SkipGram,tfidf
from src.sequence import LinearChainCRF, get_spans, BiLSTMTagger, collate,conlleval_scores
from src.transformer import ScaledDotProductAttention,TopicTransformer,SinusoidalPositionEncoding, MultiHeadSelfAttention

class TestComponents(unittest.TestCase):
 def test_lexicon_gazetteer(self):
    _,counts=build_lexicon()
    self.assertTrue(all(counts[tag]>=200 for tag in ('NOUN','VERB','ADJ')))
    g=gazetteer();self.assertTrue(len(g)>=130);self.assertEqual(len(POS_TAGS),12)
    tokens='imran khan visited islamabad'.split();tags=ner_bio(tokens,g)
    self.assertEqual(tags[:2],['B-PER','I-PER']);self.assertEqual(tags[-1],'B-LOC')
 def test_tfidf_formula_and_ppmi(self):
    docs=[['a','b','a'],['b','c'],['a','c']]
    vocab={'<pad>':0,'<unk>':1,'a':2,'b':3,'c':4}
    with tempfile.TemporaryDirectory() as temp:
     root=Path(temp)
     with patch('src.embeddings.EMB',root),patch('src.embeddings.save_json',lambda *args:None):
      tfidf([{'tokens':d,'topic':'Politics'} for d in docs],vocab)
      dense=np.load(root/'tfidf_matrix.npy')
      self.assertAlmostEqual(dense[0,2],2*np.log(3/3),places=6)
      self.assertAlmostEqual(dense[0,3],np.log(3/3),places=6)
      ppmi=ppmi_matrix(docs,vocab)
      self.assertTrue(np.isfinite(ppmi.data).all())
      self.assertTrue((ppmi.data>=0).all())
      self.assertEqual(ppmi.shape,(5,5))
      self.assertTrue((root/'ppmi_matrix.npy').exists())
 def test_skipgram_bce(self):
    m=SkipGram(20,100)
    z=m(torch.tensor([3,4]),torch.tensor([6,7]),torch.randint(1,19,(2,10)))
    self.assertTrue(torch.isfinite(z));z.backward()
    self.assertIsNotNone(m.V.weight.grad)
 def test_crf_matches_brute(self):
    torch.manual_seed(3)
    crf=LinearChainCRF(3)
    em=torch.randn(1,4,3)
    mask=torch.ones(1,4,dtype=torch.bool)
    g=torch.tensor([[0,2,1,2]])
    logz=crf.log_partition(em,mask).item()
    scores=[]
    for tup in itertools.product(range(3),repeat=4):
      score=crf.start[tup[0]]+em[0,0,tup[0]]
      for t in range(1,4):score=score+crf.transitions[tup[t-1],tup[t]]+em[0,t,tup[t]]
      score=score+crf.end[tup[-1]]
      scores.append((score.item(),tup))
    expected=torch.logsumexp(torch.tensor([x[0] for x in scores]),dim=0).item()
    self.assertAlmostEqual(logz,expected,places=5)
    want=max(scores)[1];got=tuple(crf.viterbi(em,mask)[0]);self.assertEqual(got,want)
    crf(em,g,mask).backward();self.assertIsNotNone(crf.transitions.grad)
 def test_bio_span_matching(self):
    true=[['B-PER','I-PER','O','B-LOC']]
    pred=[['B-PER','I-PER','O','O']]
    scores,fp,fn=conlleval_scores(true,pred)
    self.assertEqual(scores['PER']['TP'],1);self.assertEqual(scores['LOC']['FN'],1)
    self.assertEqual(len(fn),1)
 def test_attn_mask_and_cls(self):
    att=ScaledDotProductAttention()
    x=torch.randn(2,6,32)
    out,w=att(x,x,x,torch.tensor([[1,1,1,0,0,0],[1,1,1,1,0,0]],dtype=torch.bool))
    self.assertEqual(out.shape,(2,6,32));self.assertLess(w[0,:,-1].max().item(),1e-8)
    self.assertTrue(torch.allclose(w.sum(-1),torch.ones(2,6),atol=1e-5))
    self.assertFalse(any('pe' in n for n,_ in SinusoidalPositionEncoding().named_parameters()))
    model=TopicTransformer(70)
    tok=torch.randint(1,69,(1,8));tok[0,-2:]=0
    res,weights=model(tok,tok.ne(0),return_attention=True)
    self.assertEqual(res.shape,(1,5));self.assertEqual(weights.shape,(1,4,9,9))
    self.assertEqual(len(model.blocks),4)
 def test_split(self):
    labels=['Politics']*80+['Sports']*80+['Economy']*80+['International']*80+['Health & Society']*80
    splits=stratified_split(list(range(400)),labels)
    self.assertEqual([len(s) for s in splits],[280,60,60])
    self.assertEqual(set(splits[0])&set(splits[2]),set())

if __name__=='__main__':unittest.main()