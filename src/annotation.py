"""Part 2 annotation: transparent rule-based silver labels, manually curated seeds.
Manually spot-check generated CoNLL files; no human-gold labels are fabricated.
"""
from __future__ import annotations
import re, random, json
from collections import Counter,defaultdict
from pathlib import Path
from .common import DATA, RES, tokenize, normalize, load_articles, stratified_split, show_counts, save_json, SENT_RE
from .a1_linguistics import urdu_stem, urdu_lemmatize

# The hand-curated root words below are intentionally readable and editable; the
# script adds common morphological variants to increase coverage, not pretrained NLP.
NOUN_ROOTS='''
city village country nation province district region street market house home room door window table chair school college university student teacher classroom book notebook pen pencil paper class lesson exam result degree education knowledge language word sentence story article news newspaper journalist report interview camera photograph video television radio internet website computer laptop phone mobile network signal software hardware application program algorithm model data dataset database file folder server system device machine robot science research study experiment theory method project assignment lecture library laboratory office company business industry factory shop store customer employee employer manager director president minister leader government parliament assembly election vote party policy law constitution court judge lawyer police army soldier officer border airport railway train bus car truck road bridge river canal lake sea mountain valley forest park garden tree flower fruit vegetable crop farmer agriculture wheat rice cotton sugar milk water food bread meat fish egg tea coffee breakfast lunch dinner restaurant hotel kitchen family father mother brother sister son daughter child children man woman person people friend neighbour community society culture tradition festival holiday health hospital clinic patient doctor nurse medicine vaccine disease infection treatment surgery blood heart brain body hand foot eye ear nose face hair skin bone muscle sport sports match team cricket football hockey tennis badminton player coach captain wicket ball bat stadium referee goal score tournament league champion championship win loss victory defeat economy bank money currency price tax trade commerce import export budget inflation salary wage income expense investment finance company market industry product service technology transport environment climate weather rain flood storm earthquake temperature energy power electricity gas oil petrol engine fuel security safety peace conflict war treaty diplomacy embassy country capital foreign affairs global world continent island geography religion mosque temple church school doctor lawyer engineer journalist writer poet artist actor singer film music poetry literature museum history'''.split()
VERB_ROOTS='''
be have do say go get make know think take see come want look use find give tell work call try ask need feel become leave put mean keep let begin seem help talk turn start show hear play run move live believe bring happen write provide sit stand lose pay meet include continue set learn change lead understand watch follow stop create speak read allow add spend grow open walk win offer remember love consider appear buy wait serve die send expect build stay fall cut reach kill remain suggest raise pass sell require report decide pull return explain hope develop carry break receive agree support hit produce eat cover catch draw choose cause point listen realize place close involve increase decrease reduce improve prepare train evaluate test compute encode decode classify label predict compare analyze plot save load print transform normalize initialize update optimize calculate backpropagate propagate transfer communicate infer inferencing generate finish publish discuss describe identify measure count check verify review revise edit submit download upload connect commit push run execute compile debug implement construct split sample select annotate assign score tag name align reconstruct process monitor inspect interpret learn teach study focus complete schedule plan organize evaluate compare determine record collect clean preprocess stem lemmatize tokenize filter map count select average summarize join merge repeat shuffle reset return forward backward backwarding mask pad truncate initialize tune freeze unfreeze inject attend attendable attendable translate translateable search retrieve extract compare correlate rank discover investigate observe state establish confirm distinguish consider conclude hypothesize model organize collaborate coordinate manage mediate authorize validate navigate draw calculate estimate measure compute multiply divide subtract add convert travel walk drive arrive depart visit call vote elect train qualify play bat bowl win lose score draw compete match serve treat heal vaccinate nurse diagnose fight negotiate import export manufacture trade purchase sell pay hire fire teach learn graduate educate grow harvest sow cook drink wash clean build design program deploy configure evaluate deploy connect pair compare summarize discuss report draft finalize write illustrate interpret visualize detect discover classify detect predict classify inspect select assign type annotate mask compute optimize test develop design improve reduce shorten accelerate accelerate achieve finish'''.split()
ADJ_ROOTS='''
good bad new old young large small big tiny huge long short wide narrow high low deep shallow tall brief fast slow quick rapid early late easy hard difficult simple complex clear unclear correct incorrect right wrong true false real fake authentic modern ancient current recent previous next first final last initial original official public private common rare frequent unusual unique normal special important significant minor major basic advanced powerful weak strong healthy sick dangerous safe secure insecure open closed active passive busy free poor rich wealthy expensive cheap affordable valuable economic financial political social cultural national international local foreign global domestic rural urban public private legal illegal medical clinical scientific technical digital virtual physical natural artificial human automated manual efficient inefficient accurate inaccurate precise vague robust fragile stable unstable fixed variable random deterministic fair unfair unbiased biased positive negative good better best poor poorer worst reliable unreliable responsible irresponsible ethical unethical intelligent smart creative innovative conventional traditional general specific particular detailed comprehensive incomplete complete helpful useful useless difficult challenging ambitious practical theoretical experimental analytical empirical semantic syntactic lexical grammatical contextual relational meaningful meaningless relevant irrelevant unrelated independent dependent interdependent visible invisible transparent opaque noisy clean messy dirty smooth rough soft hard light dark bright dull calm angry happy sad excited upset worried serious funny formal informal casual respectful polite friendly hostile supportive critical skeptical optimistic pessimistic objective subjective clear concise verbose brief long informative descriptive excellent average normal odd early late immediate delayed delayed available unavailable successful unsuccessful confident uncertain effective ineffective predictive generative realistic unrealistic strong weak strong patient impatient original copied human academic professional educational linguistic computational statistical neural hierarchical distributed sequential recurrent bidirectional unidirectional frozen trainable learned learnable generalizable accurate precise distinct different similar identical consistent inconsistent novel complex simple deep shallow human social government governmental urban rural regional federal provincial municipal parliamentary electoral economic industrial agricultural environmental geological geographical demographic historical historical ancient medieval religious medical biological chemical mathematical logical approximate exact observed measured reported tested valid invalid trained untrained significant small medium large black white green red blue yellow brown golden silver loud quiet bright dark rainy sunny cloudy windy foggy severe mild temporary permanent mobile stationary independent international healthy wealthy mature minor adult young elderly digital electronic mechanical visual textual spoken written conversational structured unstructured linked connected disconnected efficient trustworthy honest dishonest genuine fabricated'''.split()
ADV_WORDS='''quickly slowly carefully clearly correctly incorrectly fully partly often never always sometimes usually rarely frequently recently currently previously immediately finally early late here there now then today tomorrow yesterday very quite rather almost already still only also even especially extremely significantly strongly weakly approximately exactly relatively directly indirectly naturally manually automatically independently jointly closely differently similarly gradually suddenly rapidly safely openly honestly truly frankly mostly mainly generally specifically locally globally nationally internationally publicly privately actively passively confidently critically analytically practically easily hardly nicely badly well together apart ahead behind above below upward downward forward backward inside outside everywhere anywhere nowhere perhaps maybe probably certainly apparently likely surely absolutely strongly greatly rapidly simply directly respectively subsequently primarily secondarily automatically consistently independently thoroughly continuously steadily increasingly sharply slightly moderately nearly barely completely partly annually monthly weekly daily hourly sometimes ultimately therefore however accordingly additionally moreover consequently respectively even yet furthermore similarly differently clearly silently loudly exactly roughly gradually quickly slowly'''.split()
PRON_WORDS='''i me my mine you your yours he him his she her hers it its we us our ours they them their theirs who whom whose which what whoever myself yourself himself herself ourselves themselves mein main mujhe mera meri hum ham hamara tum tumhara aap aapka woh wo yeh ye un in kisi koi sab khud میں مجھے میرا میری ہم ہمارا تم تمہارا آپ وہ یہ ان اُن اسے انہیں کوئی کسی خود'''.split()
DET_WORDS='''a an the this that these those each every some any no few many much several all both either neither one another other such yeh wo ye woh is us iss uss ek koi kuch har tamam چند کئی کوئی یہ وہ اس ایک ایکہ'''.split()
CONJ_WORDS='''and or but nor yet so because although whereas while if unless when since after before until as whether neither either aur ya lekin magar kyunki agar jab tak پھر جبکہ اور یا لیکن مگر کیونکہ اگر جب تاکہ'''.split()
POST_WORDS='''in on at to from for with about under over above below between among within without during before after through across behind near beside into onto off up down by of per via than tak se ko mein main par pe ka ki ke kay kayliay saath baad pehle liye کے کی کا کو سے میں پر تک ساتھ بعد پہلے لیے'''.split()
PROPN_WORDS='''pakistan india karachi islamabad lahore peshawar quetta punjab sindh kpk balochistan asia china afghanistan pakistan india un islamabad karachi پنجاب سندھ پاکستان بھارت لاہور کراچی اسلام آباد پشاور کوئٹہ'''.split()

# The hand-crafted lexicon has >200 entries for each open class NOUN/VERB/ADJ
# after conservative surface expansions. The corpus may be Urdu script, for
# which the user must extend A1-specific lexicon before claiming quality.
# Manually selected Urdu vocabulary for the actual BBC Urdu script. Assignment 1
# removes Latin script, so an English-only lexicon would label nearly all tokens UNK.
UR_NOUNS="""پاکستان حکومت عدالت ریاست ملک شہر ضلع گاؤں صوبہ اسمبلی پارلیمان صدر وزیراعظم وزیر سیاسی جماعت پارٹی انتخاب ووٹ شہری عوام خاتون خواتین مرد بچہ بچے خاندان والد والدہ بیٹا بیٹی بھائی بہن والدین نوجوان بزرگ استاد طالبعلم طالبہ تعلیم سکول کالج یونیورسٹی امتحان کتاب کلاس سبق کورس تحقیق سائنس ٹیکنالوجی کمپیوٹر موبائل انٹرنیٹ مشین روبوٹ نظام توانائی بجلی گیس پانی بارش سیلاب موسم ہوا درجہ حرارت زلزلہ آفت نقصان امداد علاقے علاقہ گلی سڑک راستہ پولیس فوج سپاہی افسر شہری ہسپتال صحت ڈاکٹر مریض بیماری دوا علاج ویکسین وائرس طبی کارکن خبر رپورٹ صحافی ٹی وی اخبار میڈیا پروگرام ویڈیو تصویر ویب سائٹ دفتر ادارہ عدالت جج وکیل مقدمہ فیصلہ قانون جرم سزا حفاظت احتجاج مظاہرہ احتجاجی بجٹ معیشت تجارت کاروبار سرمایہ قیمت پیسہ روپے ڈالر ٹیکس بینک مارکیٹ صنعت زراعت فصل کسان گندم چاول سبزی گوشت روٹی سفر جہاز ٹرین ریلوے گاڑی بس موٹر سائیکل ٹریفک حادثہ کرکٹ میچ کھیل کھلاڑی ٹیم کپتان گیند بلے باز وکٹ رن فتح شکست ٹورنامنٹ فٹبال گول میدان اسٹیڈیم جنگ امن مذاکرات تنازع سرحد فوجی حکومتیں ممالک مذاکراتی معاہدہ دنیا خطہ براعظم امریکا بھارت چین افغانستان ایران سعودی عرب برطانیہ پاکستانیت ثقافت زبان فن فنکار فلم ڈرامہ گلوکار موسیقی تقریب جشن عید رمضان ریاضت کام روزگار ملازمت تنخواہ محنت خاندان معاشرہ آبادی مردم شماری تاریخ زمانہ مستقبل مسئلہ حل تبدیلی منصوبہ ترقی منصوبہ بندی نتیجہ سوال جواب وجہ اثر تحقیق خبرنامہ شعبہ تعلیمات ضروریات""".split()
UR_VERBS="""کر ہو بن جا آ چل دیکھ سن بول کہ لکھ پڑھ سیکھ سکھا بتا پوچھ سمجھ سوچ جان مان مانگ رکھ اٹھ بیٹھ دے لے لا بھیج بھیجوا پہنچ کھول بند شروع ختم روک چھوڑ بدل رہ اٹھا گرا ڈھونڈ نکال تلاش بنا بت بنا بڑھ کم سنبھال سنبھل شروع کروا ختم کروا سنبھال سنبھلو پہنچا واپس آ واپس جا مل پھیل بچ بچا مار قتل روک پکڑ گرفتار حملہ کر اطلاع دے اعلان کر جاری کر قبول کر رد کر نظر آ محسوس کر استعمال کر تیار کر جمع کر منتخب کر ہار جیت کھیل حصہ لے کام کر کام کرا مؤخر کر تجویز دے مانگ مطالبہ کر پیش کر تحقیق کر ثابت کر ظاہر کر طے کر بنا بس گزار چلا بچاؤ کر خبر دے پہنچا لکھوا جمع کروا یاد کر قبول کرا تبادلہ کر رابطہ کر ترقی کر سفر کر پڑھا سونپ متعارف کرا شروع ہو ختم ہو پڑھنے سیکھنے بولنے کرنے ہونے جانے آنے رکھنے لینے دینے چھپنے ماننے بڑھنے گرنے بڑھا کم ہوا گر گئے کر لیا کی گئی کہا گیا کہتی کہتا کرتا کرتے کرتا رہی رہا رہے موجود ہے ہیں تھا تھے ہوگی ہوگا گیا گئی گئے کریں کرے کرتا کرلیں کرو""".split()
UR_ADJS="""بڑا چھوٹا لمبا مختصر نیا پرانا اچھا برا بہتر بہترین مشکل آسان اہم ضروری قومی مقامی صوبائی وفاقی عالمی بین الاقوامی غیر ملکی ملکی پاکستانی بھارتی چینی امریکی افغان ایرانی سعودی برطانوی سرکاری نجی سیاسی معاشی اقتصادی تجارتی زرعی سماجی معاشرتی تعلیمی طبی صحت مند بیمار علمی جدید قدیم تاریخی موجودہ سابق آئندہ اگلا پچھلا حالیہ پہلا آخری دوسرا تیسرا کم زیادہ بہت فوری دیرینہ عارضی مستقل صاف گندا گرم ٹھنڈا سرد خشک گیلا طویل مضبوط کمزور تیز سست مہنگا سستا مستحکم غیر مستحکم کامیاب ناکام مناسب غیر مناسب مثبت منفی واضح مبہم سنجیدہ عام خاص حقیقی فرضی محتاط سخت نرم خوش غمگین غریب امیر نایاب مقبول معروف مشہور غیر معروف قابل ناممکن ممکن خوبصورت بدصورت دلچسپ قابل ذکر قابل اعتماد معتبر اہم ترین مختلف یکساں الگ مشترکہ شدید معمولی حساس محفوظ غیر محفوظ متاثرہ ہلاک زخمی زندہ طویل مدت قلیل مدت انسانی قدرتی موسمی ماحولیاتی معاشرتی انتظامی قانونی عدالتی مالی تکنیکی ڈیجیٹل سائنسی جدید ترین بنیادی اضافی اصلی غلط درست سادہ پیچیدہ ترقی یافتہ غیر ترقی یافتہ شمالی جنوبی مشرقی مغربی چھوٹے بڑے مضبوط تر آزاد مقید کھلا بند محفوظ ترین مؤثر موثر سرگرم غیر فعال آباد غیر آباد پرامن پُرامن صحت مند خطرناک نارمل معمول کا ذہین فعال بنیادی کم ترین زیادہ ترین""".split()
UR_ADVS="""اب آج کل پہلے بعد پھر فوری جلدی آہستہ تیزی سے یہاں وہاں کبھی اکثر ہمیشہ دوبارہ مسلسل یکدم بالکل تقریباً ممکنہ طور پر خاص طور پر عام طور پر معمولاً فی الحال رات دن اسی دوران اس دوران فوراً کبھی کبھار شاید یقیناً نمایاں طور پر عموماً زیادہ تر کم از کم اس لیے البتہ مزید صرف بہت بھی ابھی کل گزشتہ فوری طور پر""".split()


def build_lexicon():
    lex={}
    base=[('NOUN',NOUN_ROOTS),('VERB',VERB_ROOTS),('ADJ',ADJ_ROOTS),
          ('ADV',ADV_WORDS),('PRON',PRON_WORDS),('DET',DET_WORDS),
          ('CONJ',CONJ_WORDS),('POST',POST_WORDS),('PROPN',PROPN_WORDS)]
    for tag,words in base:
        for w in words:lex[normalize(w)]=tag
    for tag,words in [('NOUN',UR_NOUNS),('VERB',UR_VERBS),('ADJ',UR_ADJS),('ADV',UR_ADVS)]:
        for w in words:
            norm=normalize(w)
            lex[norm]=tag
            # A1 corpus already contains stemmed forms, so use its real rules.
            for candidate in (urdu_stem(norm),urdu_lemmatize(norm)):
                lex.setdefault(normalize(candidate),tag)
            # Transparent, programmatic inflectional expansions from curated Urdu roots.
            if ' ' not in norm:
                suffixes= ('وں','یں','ات') if tag=='NOUN' else ('نا','تا','تی','تے') if tag=='VERB' else ('ی','ے') if tag=='ADJ' else ()
                for aff in suffixes:lex.setdefault(norm+aff,tag)
    for root in NOUN_ROOTS:
        if re.fullmatch('[a-z]+',root):
            for v in [root+'s',root+'es']:lex.setdefault(v,'NOUN')
    for root in VERB_ROOTS:
        if re.fullmatch('[a-z]+',root):
            for v in [root+'s',root+'ed',root+'ing']:lex.setdefault(v,'VERB')
    for root in ADJ_ROOTS:
        if re.fullmatch('[a-z]+',root):
            for v in [root+'er',root+'est',root+'ly']:lex.setdefault(v,'ADJ' if not v.endswith('ly') else 'ADV')
    for w in [str(i) for i in range(1000)]:lex[w]='NUM'
    counts=Counter(lex.values())
    for tag in ['NOUN','VERB','ADJ']:
        if counts[tag]<200:raise RuntimeError(f'Lexicon size for {tag} is {counts[tag]}, expected >= 200')
    return lex,dict(counts)

# Seed gazetteer: 50+ Pakistani persons, 50+ locations and 30+ organizations.
PERSONS='''
Muhammad Ali Jinnah|Allama Iqbal|Liaquat Ali Khan|Fatima Jinnah|Benazir Bhutto|Zulfikar Ali Bhutto|Nawaz Sharif|Shehbaz Sharif|Imran Khan|Bilawal Bhutto|Asif Ali Zardari|Pervez Musharraf|Arif Alvi|Mamnoon Hussain|Yousaf Raza Gillani|Shahid Khaqan Abbasi|Raja Pervez Ashraf|Chaudhry Shujaat Hussain|Mian Muhammad Nawaz|Maryam Nawaz|Hamza Shehbaz|Khawaja Asif|Ishaq Dar|Shaukat Aziz|Abdul Sattar Edhi|Bilquis Edhi|Malala Yousafzai|Abdus Salam|Abdul Qadeer Khan|Dr Abdul Qadeer Khan|Javed Miandad|Wasim Akram|Waqar Younis|Shoaib Akhtar|Shahid Afridi|Babar Azam|Shaheen Afridi|Mohammad Rizwan|Sarfaraz Ahmed|Inzamam ul Haq|Younis Khan|Misbah ul Haq|Sana Mir|Nida Dar|Jahangir Khan|Jansher Khan|Arshad Nadeem|Naseem Shah|Fakhar Zaman|Shan Masood|Azhar Ali|Mohammad Amir|Hassan Ali|Sarfraz Ahmed|Atif Aslam|Nusrat Fateh Ali Khan|Abida Parveen|Ali Zafar|Fawad Khan|Mahira Khan|Saba Qamar|Hina Rabbani Khar|Sherry Rehman|Ahsan Iqbal|Miftah Ismail|Hafeez Shaikh|Rana Sanaullah|Murad Saeed|Asad Umar|Shaukat Tarin
'''.strip().split('|')
LOCATIONS='''
Pakistan|Islamabad|Karachi|Lahore|Rawalpindi|Peshawar|Quetta|Multan|Faisalabad|Hyderabad|Sukkur|Gujranwala|Sialkot|Bahawalpur|Sargodha|Abbottabad|Mansehra|Mardan|Swat|Mingora|Nowshera|Charsadda|Kohat|Bannu|Dera Ismail Khan|Dera Ghazi Khan|Rahim Yar Khan|Sahiwal|Okara|Kasur|Sheikhupura|Jhelum|Chakwal|Attock|Taxila|Wah Cantt|Murree|Muzaffarabad|Gilgit|Skardu|Hunza|Chitral|Dir|Kalam|Kaghan|Naran|Kashmir|Azad Kashmir|Punjab|Sindh|Balochistan|Khyber Pakhtunkhwa|Gwadar|Turbat|Khuzdar|Larkana|Nawabshah|Thatta|Tharparkar|Badin|Mirpur Khas|Rohri|Kotri|Layyah|Mianwali|Bhakkar|Hafizabad|Narowal|Vehari|Lodhran|Jhang|Toba Tek Singh|Chiniot|Pakpattan|Haripur|Hangu|Kurram|Kohistan|Upper Dir|Lower Dir|Bajaur|Khyber|Mohmand
'''.strip().split('|')
ORGS='''
National Assembly|Senate of Pakistan|Supreme Court|Supreme Court of Pakistan|High Court|Islamabad High Court|Lahore High Court|Sindh High Court|Peshawar High Court|Federal Shariat Court|Election Commission|Election Commission of Pakistan|Pakistan Army|Pakistan Air Force|Pakistan Navy|Government of Pakistan|State Bank of Pakistan|National Bank of Pakistan|United Bank Limited|Habib Bank Limited|Pakistan International Airlines|Pakistan Railways|Pakistan Cricket Board|PCB|Pakistan Super League|PSL|Higher Education Commission|HEC|National Database Registration Authority|NADRA|National Disaster Management Authority|NDMA|Federal Board of Revenue|FBR|Ministry of Finance|Ministry of Education|Ministry of Health|Ministry of Foreign Affairs|Ministry of Interior|Ministry of Defence|World Health Organization|United Nations|UNICEF|UNESCO|World Bank|International Monetary Fund|IMF|FAST NUCES|Quaid i Azam University|University of Karachi|National University of Sciences and Technology|NUST|University of Punjab|Lahore University of Management Sciences|LUMS|Pakistan Tehreek e Insaf|Pakistan Muslim League N|Pakistan Peoples Party|Dawn News|Geo News|ARY News|Radio Pakistan|Express Tribune|Edhi Foundation|Shaukat Khanum Hospital|Aga Khan University Hospital
'''.strip().split('|')
MISC='''
Eid ul Fitr|Eid ul Adha|Ramadan|Independence Day|Pakistan Day|Urdu|Islam|World Cup|Cricket World Cup|COVID 19|Corona Virus|Kashmir Day
'''.strip().split('|')

# Urdu-script seeds are necessary because cleaned.txt removes Roman/English text.

# Each proper name is a phrase; use a pipe delimiter, not a whitespace split.
PERSONS_UR='''محمد علی جناح|علامہ اقبال|لیاقت علی خان|فاطمہ جناح|بے نظیر بھٹو|ذوالفقار علی بھٹو|نواز شریف|شہباز شریف|عمران خان|بلاول بھٹو|آصف علی زرداری|پرویز مشرف|عارف علوی|ممنون حسین|یوسف رضا گیلانی|شاہد خاقان عباسی|راجہ پرویز اشرف|چوہدری شجاعت حسین|مریم نواز|حمزہ شہباز|خواجہ آصف|اسحاق ڈار|شوکت عزیز|عبدالستار ایدھی|بلقیس ایدھی|ملالہ یوسفزئی|عبدالسلام|عبدالقدیر خان|جاوید میانداد|وسیم اکرم|وقار یونس|شعیب اختر|شاہد آفریدی|بابر اعظم|شاہین آفریدی|محمد رضوان|سرفراز احمد|انضمام الحق|یونس خان|مصباح الحق|ثنا میر|ندا ڈار|جہانگیر خان|جان شیر خان|ارشد ندیم|نسیم شاہ|فخر زمان|شان مسعود|اظہر علی|محمد عامر|حسن علی|عاطف اسلم|نصرت فتح علی خان|عابدہ پروین|علی ظفر|فواد خان|ماہرہ خان|صبا قمر|حنا ربانی کھر|شیری رحمان|احسن اقبال|مفتاح اسماعیل|حفیظ شیخ|رانا ثنااللہ|مراد سعید|اسد عمر|شوکت ترین'''.split('|')
LOCATIONS_UR='''پاکستان|اسلام آباد|کراچی|لاہور|راولپنڈی|پشاور|کوئٹہ|ملتان|فیصل آباد|حیدر آباد|سکھر|گوجرانوالہ|سیالکوٹ|بہاولپور|سرگودھا|ایبٹ آباد|مانسہرہ|مردان|سوات|مینگورہ|نوشہرہ|چارسدہ|کوہاٹ|بنوں|ڈیرہ اسماعیل خان|ڈیرہ غازی خان|رحیم یار خان|ساہیوال|اوکاڑہ|قصور|شیخوپورہ|جہلم|چکوال|اٹک|ٹیکسلا|واہ کینٹ|مری|مظفر آباد|گلگت|سکردو|ہنزہ|چترال|دیر|کالام|کاغان|ناران|کشمیر|آزاد کشمیر|پنجاب|سندھ|بلوچستان|خیبر پختونخوا|گوادر|تربت|خضدار|لاڑکانہ|نواب شاہ|ٹھٹہ|تھرپارکر|بدین|میرپور خاص|روہڑی|کوٹری|میانوالی|بھکر|حافظ آباد|نارووال|وہاڑی|لودھراں|جھنگ|چنیوٹ|پاکپتن|ہری پور|ہنگو|کرم|باجوڑ|خیبر|مہمند'''.split('|')
ORGS_UR='''قومی اسمبلی|سینیٹ آف پاکستان|سپریم کورٹ|سپریم کورٹ آف پاکستان|ہائی کورٹ|اسلام آباد ہائی کورٹ|لاہور ہائی کورٹ|سندھ ہائی کورٹ|پشاور ہائی کورٹ|وفاقی شرعی عدالت|الیکشن کمیشن|الیکشن کمیشن آف پاکستان|پاکستان آرمی|پاکستان فضائیہ|پاکستان بحریہ|حکومت پاکستان|اسٹیٹ بینک آف پاکستان|نیشنل بینک آف پاکستان|یونائیٹڈ بینک|حبیب بینک|پاکستان انٹرنیشنل ایئرلائنز|پاکستان ریلوے|پاکستان کرکٹ بورڈ|پاکستان سپر لیگ|ہائر ایجوکیشن کمیشن|نادرا|نیشنل ڈیزاسٹر مینجمنٹ اتھارٹی|ایف بی آر|وزارت خزانہ|وزارت تعلیم|وزارت صحت|وزارت خارجہ|وزارت داخلہ|وزارت دفاع|عالمی ادارہ صحت|اقوام متحدہ|یونیسف|یونیسکو|عالمی بینک|آئی ایم ایف|فاسٹ یونیورسٹی|قائد اعظم یونیورسٹی|جامعہ کراچی|نسٹ|پنجاب یونیورسٹی|لمز|پاکستان تحریک انصاف|پاکستان مسلم لیگ ن|پاکستان پیپلز پارٹی|ڈان نیوز|جیو نیوز|اے آر وائی نیوز|ریڈیو پاکستان|ایکسپریس نیوز|ایدھی فاؤنڈیشن|شوکت خانم ہسپتال|آغا خان یونیورسٹی ہسپتال'''.split('|')


def gazetteer():
    """Exact and conservative stem-normalized matches for A1-cleaned Urdu.

    Keep multiword entities intact. Do not overwrite ambiguous collisions.
    Labels remain weak/silver labels, not manually verified ground truth.
    """
    assert len(PERSONS)>=50 and len(LOCATIONS)>=50 and len(ORGS)>=30
    out={}; ambiguous=set()
    def add(key, typ):
        if not key or key in ambiguous:return
        if key in out and out[key]!=typ:
            ambiguous.add(key);out.pop(key,None)
        else:out[key]=typ
    for typ,items in [('PER',PERSONS+PERSONS_UR),('LOC',LOCATIONS+LOCATIONS_UR),
                      ('ORG',ORGS+ORGS_UR),('MISC',MISC)]:
        for item in items:
            seq=tuple(normalize(x) for x in tokenize(item))
            add(seq,typ)
            # Assignment 1 clean_and_process_article stems words individually.
            add(tuple(normalize(urdu_stem(w)) for w in seq),typ)
            add(tuple(normalize(urdu_lemmatize(w)) for w in seq),typ)
    return out

def ner_bio(tokens, gz):
    tags=['O']*len(tokens)
    max_len=max(map(len,gz))
    i=0
    while i<len(tokens):
        for end in range(min(len(tokens),i+max_len),i,-1):
            cat=gz.get(tuple(tokens[i:end]))
            if cat:
                tags[i]='B-'+cat
                for j in range(i+1,end):tags[j]='I-'+cat
                i=end;break
        else:i+=1
    return tags


POS_TAGS=['NOUN','VERB','ADJ','ADV','PRON','DET','CONJ','POST','NUM','PUNC','UNK','PROPN']
NER_TAGS=['O','B-PER','I-PER','B-LOC','I-LOC','B-ORG','I-ORG','B-MISC','I-MISC']


def rule_tag(w,lex):
    if re.fullmatch(r'[^\w\s]+',w):return 'PUNC'
    if w.lower()=='<num>' or re.fullmatch(r'[\d.,]+',w):return 'NUM'
    norm=normalize(w)
    if norm in lex:return lex[norm]
    # Assignment 1's own lexicon/morphology provides an interpretable fallback.
    # The original cleaned.txt has already been stemmed: this is only a lookup aid.
    for candidate in (urdu_lemmatize(norm), urdu_stem(norm),
                      urdu_stem(urdu_lemmatize(norm))):
        if candidate in lex:return lex[candidate]
    # Remaining English morphological rules are for raw/Roman tokens, if present.
    if re.fullmatch('[a-z]+',norm):
        for suffix in ['ing','ed','es','s']:
            if norm.endswith(suffix) and len(norm)>len(suffix)+2:
                stem=norm[:-len(suffix)]
                if stem in lex and lex[stem] in ['NOUN','VERB']:return lex[stem]
        if norm.endswith('ly'):return 'ADV'
        if norm.endswith(('tion','ment','ness','ity')):return 'NOUN'
        if norm.endswith(('ous','ive','able','al')):return 'ADJ'
    return 'UNK'


def create_sentences(articles):
    out=[]
    for art in articles:
        pieces=[s.strip() for s in SENT_RE.split(art['text']) if s.strip()]
        for s in pieces:
            tokens=tokenize(s,keep_punc=True)
            if 5<=len(tokens)<=120:
                out.append({'tokens':tokens,'topic':art['topic'],'article_id':art['id']})
    return out


def write_conll(samples,split,task):
    path=DATA/f'{task}_{split}.conll'
    with path.open('w',encoding='utf-8') as f:
        for sample in samples:
            for tok,tag in zip(sample['tokens'],sample[task]): f.write(f'{tok}\t{tag}\n')
            f.write('\n')


def run():
    """Exactly 500 varied sentences; ≥100 from three topics; entity-aware sample.

    Sampling uses gazetteer hits only, never model predictions. Splits are
    made before training-only oversampling; validation/test are not resampled.
    """
    random.seed(42)
    articles=load_articles();pool=create_sentences(articles)
    lex,lc=build_lexicon();gz=gazetteer()
    for s in pool:
        s['pos']=[rule_tag(w,lex) for w in s['tokens']]
        s['ner']=ner_bio([normalize(w) for w in s['tokens']],gz)
        s['entity_count']=sum(tag.startswith('B-') for tag in s['ner'])
    grouped=defaultdict(list)
    for s in pool:grouped[s['topic']].append(s)
    eligible=[t for t,ss in grouped.items() if len(ss)>=100]
    if len(eligible)<3:raise ValueError('Three topics with >=100 sentences required')
    rng=random.Random(42);selected=[];chosen=set()
    def choose(group,count):
        available=[s for s in group if id(s) not in chosen]
        positive=[s for s in available if s['entity_count']>0]
        # ~70% entity-containing sentences, rest random background examples.
        target=min(len(positive),int(count*.70))
        pick=rng.sample(positive,target) if target else []
        used={id(s) for s in pick}
        others=[s for s in available if id(s) not in used]
        if len(others)<count-target:raise ValueError('Insufficient distinct sentences')
        pick+=rng.sample(others,count-target)
        chosen.update(id(s) for s in pick)
        return pick
    major=sorted(eligible,key=lambda t:len(grouped[t]),reverse=True)[:3]
    for t in major:selected+=choose(grouped[t],100)
    selected+=choose(pool,200)
    rng.shuffle(selected)
    assert len(selected)==500 and len(chosen)==500
    labels=[s['topic'] for s in selected]
    tr,va,te=stratified_split(selected,labels,42)
    # Avoid accidentally having no entities in a holdout; no test data used
    # for learning or hyperparameter search.
    for split,ids in [('train',tr),('val',va),('test',te)]:
        chunk=[selected[i] for i in ids]
        write_conll(chunk,split,'pos');write_conll(chunk,split,'ner')
        print('POS',split,dict(Counter(t for s in chunk for t in s['pos'])))
        print('NER',split,dict(Counter(t for s in chunk for t in s['ner'])))
        print('NER entity spans',split,sum(s['entity_count'] for s in chunk))
    audits=[{'tokens':s['tokens'],'POS':s['pos'],'NER':s['ner'],
             'topic':s['topic'],'entity_count':s['entity_count']} for s in selected[:30]]
    save_json('annotation_review_samples.json',audits)
    summary={'sentences':500,'split_sizes':{k:len(v) for k,v in [('train',tr),('val',va),('test',te)]},
             'topics':dict(Counter(labels)),'lexicon_counts':lc,'gazetteer_unique_entries':len(gz),
             'entity_spans_by_split':{k:sum(selected[i]['entity_count'] for i in ids)
                 for k,ids in [('train',tr),('val',va),('test',te)]},
             'gazetteer_seed_counts':{'PER':len(PERSONS)+len(PERSONS_UR),
                 'LOC':len(LOCATIONS)+len(LOCATIONS_UR),'ORG':len(ORGS)+len(ORGS_UR),'MISC':len(MISC)},
             'label_warning':'Weak gazetteer labels; manual expert annotation still required for gold accuracy.',
             'sampling_note':'70% of each selection stratum selected by gazetteer-hit status; no augmentation after split.',
             'pos_tag_note':'PDF says 12 POS tags but lists 11; PROPN included as 12th.'}
    save_json('annotation_summary.json',summary)
    return summary

if __name__=='__main__':run()
