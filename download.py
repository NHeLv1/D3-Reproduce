#数据集下载
from modelscope.msdatasets import MsDataset
ds =  MsDataset.load('cccnju/GenVideo-100K', cache_dir='./GenVideo-100K')
#您可按需配置 subset_name、split，参照“快速使用”示例代码