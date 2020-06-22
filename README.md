# 2020腾讯广告算法大赛

## version v1( lsh, path=./ipynb)
* [数据序列化处理](#数据序列化处理)
* [训练word2vec词向量](#训练word2vec词向量)
* [训练模型](#训练模型)


<a id='数据序列化处理'></a>
### 数据序列化处理
* 需要如下目录和文件
```
|--tencent-ad-amp
  |--w2v_trainer.ipynb
  |--train_word_vector.py
  |--prosess_data.py
  |--train_artifact
    |--user.csv
    |--click_log.csv
    |--ad.csv
  |--test_artifact
    |--click_log.csv
    |--ad.csv
```
将w2v_trainer.ipynb运行至如下命令行：
```python
    logger = initiate_logger('input_generate.log')
    generate(logger=logger) 
```
即可将数据处理成序列形式的文本
<a id='训练word2vec词向量'></a>
### 训练word2vec词向量
* 需要如下目录和文件
```
|--tencent-ad-amp
  |--w2v_trainer.ipynb
  |--train_word_vector.py
  |--prosess_data.py
  |--input_artifact
    |--test_ad_id_seq.pkl
    |--test_advertiser_id_seq.pkl
    |--test_creative_id_seq.pkl
    |--test_idx_shuffle.npy
    |--test_industry_id_seq.pkl
    |--test_product_category_id_seq.pkl
    |--test_product_id_seq.pkl
    |--train_ad_id_seq.pkl
    |--train_advertiser_id_seq.pkl
    |--train_age.npy
    |--train_creative_id_seq.pkl
    |--train_gender.npy
    |--train_idx_shuffle.npy
    |--train_industry_id_seq.pkl
    |--train_product_category_id_seq.pkl
    |--train_product_id_seq.pkl
```
将w2v_trainer.ipynb运行至如下命令行：
```python
    i = 0
    for target in corpus_dic.keys():
        ebd_dim_list = [128,128,128,128,64,128]
        train_wor2vec(target,ebd_dim_list[i])
        i+=1
```
即可将完成词向量训练

<a id='训练模型'></a>
### 训练模型
* 需要如下目录和文件
```
|--tencent-ad-amp
  |--lstm_attention.ipynb
  |--input_artifact
    |--test_ad_id_seq.pkl
    |--test_advertiser_id_seq.pkl
    |--test_creative_id_seq.pkl
    |--test_idx_shuffle.npy
    |--test_industry_id_seq.pkl
    |--test_product_category_id_seq.pkl
    |--test_product_id_seq.pkl
    |--train_ad_id_seq.pkl
    |--train_advertiser_id_seq.pkl
    |--train_age.npy
    |--train_creative_id_seq.pkl
    |--train_gender.npy
    |--train_idx_shuffle.npy
    |--train_industry_id_seq.pkl
    |--train_product_category_id_seq.pkl
    |--train_product_id_seq.pkl
  |--embed_artifact
    |--ad_sg_embed_s128_uid8eo1c.wv.vectors.npy
    |--ad_sg_embed_s128_wv_gdyyb1qc
    |--ad_sg_embed_s128_wv_gdyyb1qc.vectors.npy
    |--advertiser_sg_embed_s128_wv_t70t5_xl
    |--creative_sg_embed_s128_wv_s0oc73kx
    |--creative_sg_embed_s128_wv_s0oc73kx.vectors.npy
    |--embed_train_ad_id_seq.pkl
    |--embed_train_advertiser_id_seq.pkl
    |--embed_train_creative_id_seq.pkl
    |--embed_train_industry_id_seq.pkl
    |--embed_train_product_category_id_seq.pkl
    |--embed_train_product_id_seq.pkl
    |--industry_sg_embed_s64_wv_bg7iidgr
    |--product_sg_embed_s128_wv_dyplgj8j
    |--w2v_registry.json
    |--wv_registry.json
```
##### 运行lstm_attention.ipynb
由于时间仓促，没有进行K折交叉训练
该无手动特征输入，模型深度不够，效果不理想


## version v2(lgz, path=./py)
* [分割数据集](#数据预处理与划分)
* [训练word2vec](#训练word2vec词向量)
* [训练模型](#训练模型)

<a id='数据预处理与划分'></a>
###	数据预处理和划分
````
python3 input_generate.py
python3 input_split.py
````
    
	
运行之后在根目录下出现以下文件
````python
 |--input_artifact
    |--train_idx_shuffle.npy
    |--train_age.npy
    |--train_gender.npy
    |--train_creative_id_seq.pkl
    |--train_ad_id_seq.pkl
    |--train_advertiser_id_seq.pkl
    |--train_product_id_seq.pkl
    |--test_idx_shuffle.npy
    |--test_creative_id_seq.pkl
    |--test_ad_id_seq.pkl
    |--test_advertiser_id_seq.pkl
    |--test_product_id_seq.pkl
  |--embed_artifact
    |--embed_train_creative_id_seq.pkl
    |--embed_train_ad_id_seq.pkl
    |--embed_train_advertiser_id_seq.pkl
    |--embed_train_product_id_seq.pkl
  |--model_artifact
  |--output_artifact
  |--train_artifact
  |--test_artifact
````
<a id='训练word2vec词向量'></a>
###	训练word2vec词向量
```python
python3 train_w2v.py creative 128
python3 train_w2v.py ad 128
python3 train_w2v.py advertiser 128
python3 train_w2v.py product 128
python3 train_w2v.py industry 64
python3 train_w2v.py product_category 64
```
<a id='训练模型'></a>
### 训练模型
````python
python3 train_age_gru.py 20 512 100 1e-3
python3 train_gender_lstm.py 10 512 100 1e-3
````

