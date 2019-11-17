# Discompile .pyc file

- README说用的是python 3.5，实际上pyc字节码文件是用python 3.6生成的

# Install dependence

```
pip install -r requirements.txt
```

- 中途可能有报错，不断重试即可

# Run program

## 先用官方命令

```
usage: main.py [-h] model dataset input_file output_file

D3NER Program: Recognize biomedical entities in text documents.

positional arguments:
    model        the name of the model being used, i.e: d3ner_cdr
    dataset      the name of the dataset that the model was trained on, i.e: cdr
    input_file   path to the input file, i.e: data/cdr/cdr_test.txt
    output_file  path to the output file, i.e: output.txt

optional arguments:
    -h, --help   show this help message and exit
```

- python main.py d3ner_cdr cdr data/cdr/cdr_test.txt output.txt

### 运行前的准备工作

#### 这里用的 input_file 是 cdr_test.txt，打开来一看发现看不懂

##### <a name="cdr-test-txt-file-section"></a> cdr_test.txt 文件节选
```
8701013|t|Famotidine-associated delirium. A series of six cases. (文章标题)
8701013|a|Famotidine is a histamine H2-receptor antagonist used in inpatient settings for prevention of stress ulcers and is showing increasing popularity because of its low cost. Although all of the currently available H2-receptor antagonists have shown the propensity to cause delirium, only two previously reported cases have been associated with famotidine. The authors report on six cases of famotidine-associated delirium in hospitalized patients who cleared completely upon removal of famotidine. The pharmacokinetics of famotidine are reviewed, with no change in its metabolism in the elderly population seen. The implications of using famotidine in elderly persons are discussed. （文章摘要）
（文章序号）（实体名称在文中出现的位置）（实体名称）（实体类别）（实体编号）
8701013	0	10	Famotidine	Chemical	D015738
8701013	22	30	delirium	Disease	D003693
8701013	55	65	Famotidine	Chemical	D015738
8701013	156	162	ulcers	Disease	D014456
8701013	324	332	delirium	Disease	D003693
8701013	395	405	famotidine	Chemical	D015738
8701013	442	452	famotidine	Chemical	D015738
8701013	464	472	delirium	Disease	D003693
8701013	537	547	famotidine	Chemical	D015738
8701013	573	583	famotidine	Chemical	D015738
8701013	689	699	famotidine	Chemical	D015738
（文章序号）（CID即Chemical-Induced-Disease）（化学品实体序号）（疾病实体序号）
8701013	CID	D015738	D003693
```

- 论文里有关 cdr 的说明

> We evaluate D3NER on three **benchmark corpora** of disease, chemical and gene/protein annotations: **the BioCreative V Chemical Disease Relation (BC5 CDR) corpus** (Li et al., 2015), the NCBI Disease corpus (Do!gan et al., 2014) (see Table 1) and FSU-PRGE (Hahn et al., 2010). 

- 太简略了，还是看不懂

- Google 什么是 the BioCreative V Chemical Disease Relation (BC5 CDR) corpus

- 找到一篇文章 **Overview of the BioCreative V Chemical Disease Relation (CDR) Task**

- 看懂了
    - cdr是个测试用的数据集
    - 用来测试以下两类模型的准确性
        1. 针对疾病和化学品的命名实体识别模型
        2. 针对"xxx化学品可诱发yyy疾病"这一实体间关系的识别模型
    - D3NER是上述的第一类模型
    - 所有出现在cdr中的有关疾病和化学品的实体名称都出自MeSH（可以看作是一个生物医学领域的专用术语词典）
    - cdr分为三个子集：测试集（test）、训练集（train）、开发集（dev）
    - 每个子集各包括500篇生物医学文章（的摘要）
    - 上面[cdr_test.txt 文件节选](#cdr-test-txt-file-section)中的内容就对应着一篇文章

#### 先大致阅读一下main.py，搞清楚各个参数具体是什么意思

#### 从 main.py 开始解释执行，报错

- 原因：缺少 en_core_web_md

##### en_core_web_md 是啥

- en_core_web_md 是 spacy 的一个 language model

- en-针对英语，md-中等大小（spacy 官网上还有一个sm（小） model，一个lg（大） model）

##### spacy 是啥

- spacy 是一个用于自然语言处理 python 库，旨在帮助用户构建基于自然语言处理的应用程序

- spacy 提供底层的文本处理功能，包括：
    - 把文本分解成单词或句子
    - 给单词标注词性
    - 标注句法成分（主语、宾语 etc）
    - etc

- 参见[spaCy 101: Everything you need to know · spaCy Usage Documentation](https://spacy.io/usage/spacy-101)

##### language model 是啥

- 是 spacy 的可选组件

- 像**标注句法成分（主语、宾语 etc）**之类的功能需要有对应的 language model

#### 安装 en_core_web_md

```
python -m spacy download en_core_web_md
```

- 报错

```
Traceback (most recent call last):
  File "C:\Program Files\Python36\lib\runpy.py", line 193, in _run_module_as_main
    "__main__", mod_spec)
  File "C:\Program Files\Python36\lib\runpy.py", line 85, in _run_code
    exec(code, run_globals)
  File "C:\Program Files\Python36\lib\site-packages\spacy\__main__.py", line 133, in <module>
    plac.Interpreter.call(CLI)
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 1142, in call
    print(out)
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 914, in __exit__
    self.close(exctype, exc, tb)
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 952, in close
    self._interpreter.throw(exctype, exc, tb)
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 964, in _make_interpreter
    arglist = yield task
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 1139, in call
    raise_(task.etype, task.exc, task.tb)
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 53, in raise_
    raise exc.with_traceback(tb)
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 380, in _wrap
    for value in genobj:
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 95, in gen_exc
    raise_(etype, exc, tb)
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 53, in raise_
    raise exc.with_traceback(tb)
  File "C:\Program Files\Python36\lib\site-packages\plac_ext.py", line 966, in _make_interpreter
    cmd, result = self.parser.consume(arglist)
  File "C:\Program Files\Python36\lib\site-packages\plac_core.py", line 207, in consume
    return cmd, self.func(*(args + varargs + extraopts), **kwargs)
  File "C:\Program Files\Python36\lib\site-packages\spacy\__main__.py", line 33, in download
    cli_download(model, direct)
  File "C:\Program Files\Python36\lib\site-packages\spacy\cli\download.py", line 20, in download
    model_name = check_shortcut(model)
  File "C:\Program Files\Python36\lib\site-packages\spacy\cli\download.py", line 39, in check_shortcut
    shortcuts = get_json(about.__shortcuts__, "available shortcuts")
  File "C:\Program Files\Python36\lib\site-packages\spacy\cli\download.py", line 28, in get_json
    r = requests.get(url)
  File "C:\Program Files\Python36\lib\site-packages\requests\api.py", line 75, in get
    return request('get', url, params=params, **kwargs)
  File "C:\Program Files\Python36\lib\site-packages\requests\api.py", line 60, in request
    return session.request(method=method, url=url, **kwargs)
  File "C:\Program Files\Python36\lib\site-packages\requests\sessions.py", line 533, in request
    resp = self.send(prep, **send_kwargs)
  File "C:\Program Files\Python36\lib\site-packages\requests\sessions.py", line 646, in send
    r = adapter.send(request, **kwargs)
  File "C:\Program Files\Python36\lib\site-packages\requests\adapters.py", line 498, in send
    raise ConnectionError(err, request=request)
requests.exceptions.ConnectionError: ('Connection aborted.', TimeoutError(10060, 'A connection attempt failed because the connected party did not properly respond after a period of time, or established connection failed because connected host has failed to respond', None, 10060, None))
```

- 貌似是网络连接问题

- 尝试从[Github](https://github.com/explosion/spacy-models/releases//tag/en_core_web_md-2.2.0)上直接下 model 

- 然后使用 pip 直接安装下到本地的 model （参见[Link](https://spacy.io/usage/models#download-pip)）

```
pip install /path/to/en_core_web_md-2.2.0.tar.gz
```

- 安装过程又不停报错，不断重试（估计是 Windows 的缘故）（最终发现是没开管理员权限）

#### 从 main.py 开始解释执行，程序加载完spacy后就卡住了

- 暂停程序执行，检查调用栈

- 发现程序中一个死循环（不清楚是否是反编译器的bug）

```
while 1:
    if i < len(raw_sentences):
        if cur_sent[(-1)] in '.!?;':
            new_sentences.append(cur_sent)
            cur_sent = raw_sentences[i]
        else:
            cur_sent += ' ' + raw_sentences[i]
        i += 1
```

- 应该改成

```
while i < len(raw_sentences):
  if cur_sent[(-1)] in '.!?;':
      new_sentences.append(cur_sent)
      cur_sent = raw_sentences[i]
  else:
      cur_sent += ' ' + raw_sentences[i]
  i += 1
```

#### 关键代码说明

- pipelines.py > class NerPipeline

```
def run(self):
    raw_documents = self.reader.read()
    """
    数据读入阶段结束，raw_documents = {
      文章序号: {
        'a': 文章摘要
        't': 文章标题
      }
    }
    """
    title_docs, abstract_docs = self.data_manager.parse_documents(raw_documents)
    title_doc_objs = pre_process.process(title_docs, self.pre_config, constants.SENTENCE_TYPE_TITLE)
    abs_doc_objs = pre_process.process(abstract_docs, self.pre_config, constants.SENTENCE_TYPE_ABSTRACT)
    doc_objects = self.data_manager.merge_documents(title_doc_objs, abs_doc_objs)
    """
    预处理阶段结束，doc_objects = [
      content: 文章内容（标题+摘要）
      id: 文章序号
      metadata: {...}
      sentences: [
        content: 句子内容
        doc_offset: (文章中起始位置, 文章中结束位置)
        metadata: {...}
        tokens: [
            content: 单词内容
            doc_offset: (文章中起始位置, 文章中结束位置)
            metadata: {...}
            processed_content: 单词内容
            sentence_offset: (句中起始位置, 句中结束位置)
        ]
      ]
    ]
    """
    dict_nern = ner.process(doc_objects, self.nern_config)
    """
    识别阶段结束，dict_ner = [
      文章序号: [
        content: 模型识别出的实体名称
        ids: {...}
        tokens: [
          ...
        ]
        type: 该实体所属的类别
      ]
    ]
    """
    self.writer.write(self.output_file, raw_documents, dict_nern)
```

#### 预处理阶段结束后又卡住了

- 暂停程序也不管用，程序停不下来，调用栈信息拿不到，不知道程序卡在哪了

- 只有反复重启、单步调试找卡住的位置

- 又是死循环

```
while 1:
  if idx < num_batch:
      X_batch = data['X'][start:start + self.batch_size]
      Y_nen_batch = data['Y_nen'][start:start + self.batch_size]
      Z_batch = data['Z'][start:start + self.batch_size]
      char_ids, word_ids = zip(*[zip(*x) for x in X_batch])
      word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0)
      char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
      nen_labels, _ = pad_sequences(Y_nen_batch, pad_tok=([0] * self.nen_label_size), nlevels=3)
      pos_ids, _ = pad_sequences(Z_batch, pad_tok=0)
      start += self.batch_size
      idx += 1
      yield (word_ids, char_ids, nen_labels, sequence_lengths, word_lengths, pos_ids)
```

- 改成

```
while idx < num_batch:
  X_batch = data['X'][start:start + self.batch_size]
  Y_nen_batch = data['Y_nen'][start:start + self.batch_size]
  Z_batch = data['Z'][start:start + self.batch_size]
  char_ids, word_ids = zip(*[zip(*x) for x in X_batch])
  word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0)
  char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
  nen_labels, _ = pad_sequences(Y_nen_batch, pad_tok=([0] * self.nen_label_size), nlevels=3)
  pos_ids, _ = pad_sequences(Z_batch, pad_tok=0)
  start += self.batch_size
  idx += 1
  yield (word_ids, char_ids, nen_labels, sequence_lengths, word_lengths, pos_ids)
```

- 检查了一遍所有文件里的形如 while 1: if xxx 的语句，改掉所有死循环

#### 再次运行程序，报错

```
module 'constants' has no attribute 'REV_ETYPE_MAP'
```

- 发现 contants.py 里根本没有定义 REV_ETYPE_MAP，只定义了一个 ETYPE_MAP （怀疑是作者不小心打错了）

- 把 REV_ETYPE_MAP 改成 ETYPE_MAP

#### 再次运行，报错

- 发现不是作者把 REV_ETYPE_MAP 打成了 ETYPE_MAP，而是忘记定义了 REV_ETYPE_MAP （即把 ETYPE_MAP 键值对调得到的新字典）

#### 再次运行，成功

- 得到输出文件 output.txt，它和输出文件 data/cdr/cdr_test.txt 的格式类似，下面是它的节选

```
23433219|t|The risk and associated factors of methamphetamine psychosis in methamphetamine-dependent patients in Malaysia. （文章标题）
23433219|a|OBJECTIVE: The objective of this study was to determine the risk of lifetime and current methamphetamine-induced psychosis in patients with methamphetamine dependence. The association between psychiatric co-morbidity and methamphetamine-induced psychosis was also studied. METHODS: This was a cross-sectional study conducted concurrently at a teaching hospital and a drug rehabilitation center in Malaysia. Patients with the diagnosis of methamphetamine based on DSM-IV were interviewed using the Mini International Neuropsychiatric Interview (M.I.N.I.) for methamphetamine-induced psychosis and other Axis I psychiatric disorders. The information on sociodemographic background and drug use history was obtained from interview or medical records. RESULTS: Of 292 subjects, 47.9% of the subjects had a past history of psychotic symptoms and 13.0% of the patients were having current psychotic symptoms. Co-morbid major depressive disorder (OR=7.18, 95 CI=2.612-19.708), bipolar disorder (OR=13.807, 95 CI=5.194-36.706), antisocial personality disorder (OR=12.619, 95 CI=6.702-23.759) and heavy methamphetamine uses were significantly associated with lifetime methamphetamine-induced psychosis after adjusted for other factors. Major depressive disorder (OR=2.870, CI=1.154-7.142) and antisocial personality disorder (OR=3.299, 95 CI=1.375-7.914) were the only factors associated with current psychosis. CONCLUSION: There was a high risk of psychosis in patients with methamphetamine dependence. It was associated with co-morbid affective disorder, antisocial personality, and heavy methamphetamine use. It is recommended that all cases of methamphetamine dependence should be screened for psychotic symptoms. （文章摘要）
（文章序号）（该实体名称在文中出现的位置）（模型识别出的实体名称）（该实体所属的类别）
23433219	35	50	methamphetamine	Chemical
23433219	51	60	psychosis	Disease
23433219	64	79	methamphetamine	Chemical
23433219	201	216	methamphetamine	Chemical
23433219	225	234	psychosis	Disease
23433219	252	267	methamphetamine	Chemical
23433219	304	315	psychiatric	Disease
23433219	333	348	methamphetamine	Chemical
23433219	357	366	psychosis	Disease
23433219	550	565	methamphetamine	Chemical
23433219	670	685	methamphetamine	Chemical
23433219	694	703	psychosis	Disease
23433219	714	718	Axis	Disease （识别错误）
23433219	721	742	psychiatric disorders	Disease
23433219	930	939	psychotic	Disease
23433219	995	1013	psychotic symptoms	Disease
23433219	1031	1050	depressive disorder	Disease
23433219	1082	1098	bipolar disorder	Disease
23433219	1132	1163	antisocial personality disorder	Disease
23433219	1206	1221	methamphetamine	Chemical
23433219	1271	1286	methamphetamine	Chemical
23433219	1295	1304	psychosis	Disease
23433219	1345	1364	depressive disorder	Disease
23433219	1396	1427	antisocial personality disorder	Disease
23433219	1504	1513	psychosis	Disease
23433219	1552	1561	psychosis	Disease
23433219	1579	1594	methamphetamine	Chemical
23433219	1640	1658	affective disorder	Disease
23433219	1660	1682	antisocial personality	Disease
23433219	1694	1709	methamphetamine	Chemical
23433219	1751	1766	methamphetamine	Chemical
23433219	1801	1810	psychotic	Disease
```

# 关键代码分析

```
def process(self, document):
    X, Z, Y_nen = self._TensorNer__parse_document_to_data(document)
    """
    数字化阶段结束
    """
    y_pred = self.model.predict_classes({'X':X,  'Z':Z,  'Y_nen':Y_nen}, self.transition_params)
    """
    模型预测阶段结束
    y_pred = [
      [
        句子1单词1可能所属的标签中得分最高的标签的序号
      ]
    ]
    """
    entities = self._TensorNer__decode_y_pred(y_pred, document)
    return entities
```

```
"""
document = {
  content: (文章内容)
  id: (文章序号)
}
"""
def __parse_document_to_data(self, document):
    X, Z, Y_nen = [], [], []
    abb = []
    tfs = None
    if document.id in self.vocab_ab3p:
        unzipped = list(zip(*self.vocab_ab3p[document.id]))
        """
        abb = [
          (该文章中的缩写单词)
        ]
        """
        abb = unzipped[0]
        """
        tfs = [
          tfs for ENTITY_TYPE Disease,
          tfs for ENTITY_TYPE Chemical
        ]
        """
        tfs = unzipped[1:]
    for s in document.sentences:
        x, z, y_nen = self._TensorNer__parse_sentence(s, abb, tfs)
        """
        x = (
          [
            Character-level embedding 值
          ],
          Token-level embedding 值
        )
        z = POS (词性) embedding 值
        y_nen = Abbreviation embedding 值
        """
        X.append(x)
        Z.append(z)
        Y_nen.append(y_nen)

    return (X, Z, Y_nen)
```

## AB3P 是啥

> All local abbreviations in an abstract are first identified using
Ab3P (Sohn et al., 2008). Then, the character-level n-gram TF-IDF vector for each abbreviation’s full form and every concept name in MeSH and FSU-PRGE (training set) are generated for measuring the pair-wise cosine similarity scores.

- [Github 仓库地址](https://github.com/ncbi-nlp/Ab3P)

## TF-IDF 是啥

> 在一份给定的文件里，词频（term frequency，tf）指的是某一个给定的词语在该文件中出现的频率

> 词频 = 该词语在文件中出现的次数 / 文件所包含的词语总数

> 逆向文件频率（inverse document frequency，idf）是一个词语普遍重要性的度量。某一特定词语的idf，可以由总文件数目除以包含该词语之文件的数目，再将得到的商取以10为底的对数得到

> 逆向文件频率 = lg(语料库中的文件总数 / 1+包含该词语的文件总数)

> TF-IDF = TF * IDF

> 如果一个单词在某一特定文件中出现的次数很多，但在其他文件中出现的次数很少，那么它很有可能是这个文件中的关键词之一，而由以上公式我们可知，该词语相对于该文件的TD-IDF值会很高。因此，我们可以使用TD-IDF来过滤掉文件中的常见词语，保留重要的关键词

```
def __parse_sentence(self, sentence, abb, tfs):
    w = []
    p = []
    n = []
    for i in range(len(sentence.tokens)):
        word = self._process_word(sentence.tokens[i].processed_content)
        """
        pos: (该单词的词性所对应的唯一识别码)
        """
        pos = self.vocab_poses[sentence.tokens[i].metadata['POS']]
        w += [word]
        p += [pos]
        if word in abb:
            idx = abb.index(word)
            n.append([tfs[k][idx] for k in range(len(constants.ENTITY_TYPES))])
        else:
            n.append([0] * len(constants.ENTITY_TYPES))

    return (w, p, n)
```

```
def _process_word(self, word):
    char_ids = []
    if self.vocab_chars is not None:
        for char in word:
            if char in self.vocab_chars:
                char_ids += [self.vocab_chars[char]]

    if word in self.vocab_words:
        word_id = self.vocab_words[word]
    else:
        word_id = self.vocab_words[self.unk]
    if self.vocab_chars is not None:
        """
        char_id: [
          (组成该单词的每一个字符所对应的唯一识别码)
        ]
        word_id: (该单词所对应的唯一识别码)
        """
        return (char_ids, word_id)
    else:
        return word_id
```

```
def predict_classes(self, data, transition_params):
    num_batch = len(data['X']) // self.batch_size + 1
    y_pred = []
    """
    一个batch同时处理32句话
    测试用的文章都比较短，只有几句话，一般一个batch就处理完了
    """
    for idx, batch in enumerate(self._next_batch_predict(data={'X':data['X'],  'Z':data['Z'],  'Y_nen':data['Y_nen']},
      num_batch=num_batch)):
        """
        words = [
          [
            句子x单词y的 Token-level embedding 值
            ...
          ]
        ]
        chars = [
          [
            [
              句子x单词y字符z的 Character-level embedding 值
            ]
          ]
        ]
        nen_labels = [
          [
            [
              句子x单词y实体类别z的 Abbreviation embedding 值
            ]
          ]
        ]
        sequence_lengths = [
          句子x所含的单词数
        ]
        word_lengths = [
          [
            句子x单词y所含的字符数
          ]
        ]
        poses = [
          [
            句子x单词y的 POS embedding 值
          ]
        ]
        """
        words, chars, nen_labels, sequence_lengths, word_lengths, poses = batch
        feed_dict = {self.word_ids: words, 
          self.char_ids: chars, 
          self.word_lengths: word_lengths, 
          self.nen_labels: nen_labels, 
          self.label_lens: sequence_lengths, 
          self.dropout_op: 1.0, 
          self.dropout_lstm: 1.0, 
          self.word_pos_ids: poses, 
          self.is_training: False}
        logits = self.session.run((self.logits), feed_dict=feed_dict)
        for logit, leng in zip(logits, sequence_lengths):
            """
            logit = 一个浮点数矩阵(每一列代表一个标签(tag))(模型输出)
            """
            logit = logit[:leng]
            """
            logit = 每个句子都对应一个浮点数矩阵，假设该句子中含有x个单词，则截取矩阵的前x行
            transition_params = 一个浮点数矩阵（假设有n个标签，则矩阵为n*n）
            """
            decode_sequence, _ = tf.contrib.crf.viterbi_decode(logit, transition_params)
            """
            decode_sequence = 该句中每个单词可能所属的n个标签中得分最高的那个标签的序号
            """
            y_pred.append(decode_sequence)

    return y_pred
```

```
def _next_batch_predict(self, data, num_batch):
    start = 0
    idx = 0
    while idx < num_batch:
        X_batch = data['X'][start:start + self.batch_size]
        Y_nen_batch = data['Y_nen'][start:start + self.batch_size]
        Z_batch = data['Z'][start:start + self.batch_size]
        char_ids, word_ids = zip(*[zip(*x) for x in X_batch])
        """
        对数字化阶段中得到的一系列embedding值做归一化处理
        （例如统一不同句子、不同单词的embedding数组的长度）
        """
        word_ids, sequence_lengths = pad_sequences(word_ids, pad_tok=0)
        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
        nen_labels, _ = pad_sequences(Y_nen_batch, pad_tok=([0] * self.nen_label_size), nlevels=3)
        pos_ids, _ = pad_sequences(Z_batch, pad_tok=0)
        start += self.batch_size
        idx += 1
        yield (word_ids, char_ids, nen_labels, sequence_lengths, word_lengths, pos_ids)
```

## NEN 是啥

> In natural language processing, entity linking, also referred to as named-entity linking (NEL),[1] named-entity disambiguation (NED), named-entity recognition and disambiguation (NERD) or named-entity normalization (NEN)[2] is the task of assigning a unique identity to entities (such as famous individuals, locations, or companies) mentioned in text. For example, given the sentence "Paris is the capital of France", the idea is to determine that "Paris" refers to the city of Paris and not to Paris Hilton or any other entity that could be referred to as "Paris". Entity linking is different from named-entity recognition (NER) in that NER identifies the occurrence of a named entity in text but it does not identify which specific entity it is (see Differences from other techniques).

> 在自然语言处理中，实体链接也称为命名实体链接（NEL），命名实体歧义消除（NED），命名实体识别和歧义消除（NERD）或命名实体规范化（NEN）。它是为文本中提到的实体（例如，著名的个人，位置或公司）分配唯一身份的任务。例如，给定句子“巴黎是法国的首都”，其想法是确定“巴黎”是指巴黎市而不是巴黎希尔顿或任何其他可以称为“巴黎”的实体。实体链接与命名实体识别（NER）的不同之处在于NER可以识别文本中是否存在命名实体，但不能标识其是哪个特定实体（请参阅与其他技术的差异）。

## Viterbi Decode 是啥

> A Viterbi decoder uses the Viterbi algorithm for decoding a bitstream that has been encoded using a convolutional code or trellis code.

> 维特比解码器使用维特比算法来解码使用卷积码或网格码编码的比特流

- [找到了MIT的一份讲义 (Lecture 9 - Viterbi Decoding of Convolutional
Codes)](http://web.mit.edu/6.02/www/f2010/handouts/lectures/L9.pdf)

## Viterbi Algorithm 是啥

> The Viterbi algorithm is a dynamic programming algorithm for finding the most likely sequence of hidden states—called the Viterbi path—that results in a sequence of observed events, especially in the context of Markov information sources and hidden Markov models (HMM).

> 维特比算法是一种DP算法，用于查找最可能的隐藏状态序列（称为维特比路径），该序列导致一系列观察到的事件。该算法在马尔可夫信息源和隐马尔可夫模型（HMM）的情况下尤其适用

## 卷积码是啥

> In telecommunication, a convolutional code is a type of error-correcting code that generates parity symbols via the sliding application of a boolean polynomial function to a data stream. The sliding application represents the 'convolution' of the encoder over the data, which gives rise to the term 'convolutional coding'. The sliding nature of the convolutional codes facilitates trellis decoding using a time-invariant trellis. Time invariant trellis decoding allows convolutional codes to be maximum-likelihood soft-decision decoded with reasonable complexity.

> 卷积码是信道编码（channel coding）技术的一种，在电信领域中，属于一种纠错码（error-correcting code）。相对于分组码，卷积码维持信道的记忆效应（memory property）。卷积码的由来，是因为输入的原始消息数据会和编码器（encoder）的冲激响应（impulse response）做卷积运算

**越来越看不懂了**

## Tensorflow 中的 Viterbi Decode 是啥

- [找到了stackoverflow上的一篇回答](https://stackoverflow.com/questions/51301061/how-to-understand-the-viterbi-decode-in-tensorflow)

```
def viterbi_decode(score, transition_params):
  """Decode the highest scoring sequence of tags outside of 
  TensorFlow.

  This should only be used at test time.

  Args:
    score: A [seq_len, num_tags] matrix of unary potentials.
    transition_params: A [num_tags, num_tags] matrix of binary potentials.

  Returns:
    viterbi: A [seq_len] list of integers containing the highest scoring tag
    indicies.
    viterbi_score: A float containing the score for the Viterbi 
    sequence.
  """
```

```
def __decode_y_pred(self, y_pred, document):
    entities = []
    for i in range(len(y_pred)):
        j = 0
        while j < len(y_pred[i]):
            e = None
            if self.all_labels[y_pred[i][j]][0] == 'U':
                e = BioEntity(etype=(constants.REV_ETYPE_MAP[self.all_labels[y_pred[i][j]][1]]), tokens=(document.sentences[i].tokens[j:j + 1]))
            elif self.all_labels[y_pred[i][j]][0] == 'B':
                l_idx = self._TensorNer__last_index(j, y_pred[i])
                if self.all_labels[y_pred[i][l_idx]][0] == 'L' and self.all_labels[y_pred[i][l_idx]][1] == self.all_labels[y_pred[i][j]][1]:
                    e = BioEntity(etype=(constants.REV_ETYPE_MAP[self.all_labels[y_pred[i][j]][1]]), tokens=(document.sentences[i].tokens[j:l_idx + 1]))
                j = l_idx
            j += 1
            if e:
                entities.append(e)
    return entities
```