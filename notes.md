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
