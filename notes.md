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
pip install /path/to/en_core_web_sm-2.2.0.tar.gz
```