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
8701013|a|Famotidine is a histamine H2-receptor antagonist used in inpatient settings for prevention of stress ulcers and is showing increasing popularity because of its low cost. Although all of the currently available H2-receptor antagonists have shown the propensity to cause delirium, only two previously reported cases have been associated with famotidine. The authors report on six cases of famotidine-associated delirium in hospitalized patients who cleared completely upon removal of famotidine. The pharmacokinetics of famotidine are reviewed, with no change in its metabolism in the elderly population seen. The implications of using famotidine in elderly persons are discussed. （文章内容）
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
    - 每个子集各包括500篇生物医学文章
    - 上面[cdr_test.txt 文件节选](#cdr-test-txt-file-section)中的内容就对应着一篇文章

#### 先大致阅读一下main.py，搞清楚各个参数具体是什么意思

