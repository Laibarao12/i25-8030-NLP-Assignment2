"""Create a 3-page evidence-based report.pdf; unfinished stages are explicitly marked."""
from __future__ import annotations
import json, os
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from .common import ROOT, RES, TOPICS


def read(name):
    path=RES/name
    return json.loads(path.read_text('utf-8')) if path.exists() else None


def font_setup():
    path=os.environ.get('TIMES_NEW_ROMAN_TTF','')
    if path and Path(path).exists():
        pdfmetrics.registerFont(TTFont('SubmissionTimes',path))
        return 'SubmissionTimes'
    return 'Times-Roman'


def fmt(v,percent=False):
    if v is None:return 'Not evaluated'
    try:return f'{float(v)*100:.2f}%' if percent else f'{float(v):.4f}'
    except (ValueError,TypeError):return str(v)


def run():
    available={
        'part1':read('four_condition_comparison.json'),
        'part2':read('part2_summary.json'),
        'part3':read('part3_summary.json'),
    }
    unfinished=[name for name,content in available.items() if content is None]
    font=font_setup()
    body=ParagraphStyle('Body',fontName=font,fontSize=12,leading=18,spaceAfter=7)
    head=ParagraphStyle('Head',fontName=font,fontSize=12,leading=18,spaceBefore=8,spaceAfter=6)
    title=ParagraphStyle('Title',fontName=font,fontSize=14,leading=20,alignment=TA_CENTER,spaceAfter=8)
    note=ParagraphStyle('Note',fontName=font,fontSize=10,leading=14,spaceAfter=5,textColor=colors.HexColor('#7f2525'))
    small=ParagraphStyle('Small',fontName=font,fontSize=10,leading=14,spaceAfter=5)
    story=[]
    def p(x,style=body):story.append(Paragraph(str(x).replace('&','&amp;'),style))
    def h(x):p('<b>'+x+'</b>',head)
    def tbl(rows,widths=None):
        t=Table([[Paragraph(str(c).replace('&','&amp;'),small) for c in row] for row in rows],colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7edf2')),
                               ('GRID',(0,0),(-1,-1),.3,colors.HexColor('#929aa0')),
                               ('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),3),
                               ('TOPPADDING',(0,0),(-1,-1),3)]))
        story.append(t);story.append(Spacer(1,6))
    p('DS 5007: NLP Assignment 2 — BBC Urdu Neural Pipeline',title)
    p('Student ID: 25i8030  |  Section: verify before submission',small)
    if unfinished:p('<b>INCOMPLETE DRAFT: real-data experiments missing ('+', '.join(unfinished)+'). Do not submit as a results report.</b>',note)
    h('Overview')
    p('This continuation of the BBC Urdu project examines whether word representations and neural sequence models preserve useful linguistic and topic information. The implementation uses the Assignment 1 corpus, a transparent rule-based annotation scheme, and PyTorch modules built without pretrained Transformer components. All reported test values below must originate from saved outputs, never illustrative numbers.')
    h('Part 1 Results — Embeddings (25 marks)')
    tf=read('tfidf_top_words.json');anal=read('analogies.json')
    if tf:
        for topic in TOPICS:
            words=', '.join(r['word'] for r in tf.get(topic,[])[:5])
            p(f'<b>{topic}:</b> {words or "No labelled articles"}.',small)
    else:p('TF-IDF topic word results: PENDING corpus-derived TF-IDF output.',small)
    if available['part1']:
        rows=[['Condition','MRR (eligible pairs)','Pairs scored']]
        for key,x in available['part1'].items():
            rows.append([key,fmt(x['MRR']),str(x['evaluable_pairs'])+'/20'])
        tbl(rows,[125,155,115])
        if anal:
            valid=[a for a in anal if a['correct_top3'] is not None]
            ncorrect=sum(a['correct_top3'] is True for a in valid)
            p(f'Analogy top-3 correctness: {ncorrect}/{len(valid)} evaluable tests; out-of-vocabulary cases are separately flagged.')
        p('The symmetric five-token-window PPMI vectors and Skip-gram embeddings were compared using the same cosine-ranking protocol. The C2 model uses raw text; C3 uses cleaned text; C4 doubles dimension from 100 to 200. Examine both pair coverage and MRR before identifying the strongest setting.')
    else:p('C1–C4 nearest neighbours, 20-pair MRR, training curves and 10 analogy outcomes: PENDING original raw.txt and cleaned.txt.')
    p('Interpretation caveat: a larger embedding dimension adds capacity, but it does not guarantee better semantic neighbours. TF-IDF is an article-level representation, while PPMI and Word2Vec estimate local lexical associations.')

    story.append(PageBreak())
    h('Part 2 Results — POS Tagging and Named Entities (25 marks)')
    ann=read('annotation_summary.json')
    if ann:
        p('Five hundred sampled sentences were stratified by topic. Split sizes: '+', '.join(f'{k}={v}' for k,v in ann['split_sizes'].items())+'.')
    else:p('Sentence annotation, label distributions and 70/15/15 split: PENDING; original corpus and metadata not provided.')
    p('POS/NER annotations start as rule-generated silver labels, not independently verified human gold. The assignment lists eleven POS names despite saying twelve; PROPN is recorded as an explicit twelfth class. Human review is necessary before treating held-out scores as a reliable estimate of tagging quality.')
    s2=available['part2']
    if s2:
        rows=[['Task / mode','Validation F1','Test F1']]
        for task in ('pos','ner'):
            for mode,stats in s2[task]['modes'].items():
                rows.append([task.upper()+' / '+mode,fmt(stats['val_f1']),fmt(stats['test_f1'])])
        tbl(rows,[135,125,125])
        pos=s2['pos']['test_metrics'];ner=s2['ner']['test_metrics']
        p(f'POS test token accuracy = {fmt(pos["accuracy"],True)}; macro-F1 = {fmt(pos["macro_f1"])}. NER strict entity-level overall P/R/F1 = {fmt(ner["overall"]["precision"])}, {fmt(ner["overall"]["recall"])}, {fmt(ner["overall"]["f1"])}.')
        p(f'NER F1 with CRF = {fmt(s2["ner_crf_vs_softmax"]["crf"])} and with softmax = {fmt(s2["ner_crf_vs_softmax"]["softmax"])}.')
        rows=[['Ablation','Test NER entity F1']]
        for name,r in s2['ablations'].items():rows.append([name,fmt(r['test_F1'])])
        tbl(rows,[235,150])
        p('A1 tests backward context; A2 tests dropout; A3 replaces the C3 initialization; and A4 removes structured CRF decoding. A difference in measured F1 does not itself establish a mechanism without examining example errors and variability.')
    else:
        p('Frozen/fine-tuned BiLSTM performance, CRF/Viterbi comparison, all four ablations and confusion/error examples: PENDING actual training.')
    p('For the three most frequently confused POS tag pairs, inspect two original sentences each. For NER, manually review at least five false positives and five false negatives, checking both boundaries and entity types.')

    story.append(PageBreak())
    h('Part 3 Results — Transformer Topic Classifier (20 marks)')
    s3=available['part3']
    if s3:
        b=s3['bilstm'];t=s3['transformer']
        tbl([['Model','Test accuracy','Test macro-F1'],
             ['BiLSTM topic classifier',fmt(b['accuracy'],True),fmt(b['macro_f1'])],
             ['4-block Transformer',fmt(t['accuracy'],True),fmt(t['macro_f1'])]],
            [170,115,110])
        p(f"Final encoder-layer attention heatmaps: {len(s3['attention_heatmaps'])} figures generated from correctly classified test articles, using two attention heads per article.")
        p('BiLSTM vs Transformer comparison (12 sentences):',small)
        for sentence in s3['comparison_sentences']:
            p(sentence,small)
    else:
        p('Five-class distribution, 20-epoch accuracy/loss curves, test confusion matrix, and six final-layer attention heatmaps: PENDING actual training.')
        p('The classifier uses independently projected four-head attention, a fixed sinusoidal positional buffer, four pre-normalized residual encoder blocks, a learned [CLS] token and a two-layer MLP classification head. A separate BiLSTM article classifier is required for a like-for-like task comparison.')
    h('Conclusion')
    if not unfinished:
        p('The results above provide an auditable comparison of sparse and dense word representations, structured and unstructured sequence decoding, and recurrent versus attention-based topic classification. Error analysis and manual review remain necessary to understand the limitations of automatically created annotations.')
    else:
        p('No test accuracy or comparison can be concluded until cleaned.txt, raw.txt and Metadata.json are supplied, models are trained, and notebook outputs are verified. This draft intentionally does not invent experimental results.')
    p('GitHub repository URL: ADD PUBLIC REPOSITORY URL AFTER PUSHING THE REAL PROJECT',small)
    target=ROOT/'report.pdf'
    doc=SimpleDocTemplate(str(target),pagesize=A4,leftMargin=44,rightMargin=44,topMargin=43,
                          bottomMargin=40,title='DS5007 Assignment 2 report'+(' — INCOMPLETE' if unfinished else ''))
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFont(font,9)
        canvas.drawRightString(A4[0]-44,23,f'DS5007 • Page {doc.page}');canvas.restoreState()
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    print('Report:',target,'incomplete stages:',unfinished)
    return target

if __name__=='__main__':run()