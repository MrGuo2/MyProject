#### 经过数据分析后发现原始数据比较混乱。首先将Train文件和Members文件进行外联合，其次将NULL值进行粗处理(将NULL数目较少变量的信息进行删除),再次将数量较少的变量进行合并分桶处理(为清理离散点)，然后将NULL较多的数据进行OneHot编码，最后将变量进行LabelEncoder编码将数据进行存储。观察数据后发现，数据中并没有太多的连续型变量，所以打算用LightGBM进行处理。首先用StandScaler对一些连续性变量进行处理，其次用基于用户的协同过滤、基于物品的协同过滤以及基于LFM的协同过滤构建特征(user-pop、item-rate、lfm-reco等5个特征)，然后将这些特征进行保存，到此特征工程已经结束。因为数据量太大，最后使用默认参数进行调优。