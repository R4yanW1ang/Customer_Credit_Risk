## PYTHON 
## ******************************************************************** ##
## author: hw_wangzirui
## create time: 2022/07/26 09:33:52 GMT+08:00
## ******************************************************************** ##
#encoding:utf-8
#!/bin/env python3
# coding=utf-8
import numpy as np
import pandas as pd
import warnings
import psycopg2
from collections import Counter
from imblearn.over_sampling import SMOTE
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_validate
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
import sklearn.metrics as mt
from urllib import parse
from sqlalchemy import create_engine
import datetime
import base64
from Crypto.Cipher import AES
import configparser
warnings.filterwarnings("ignore")

class CustomerCreditRisk():
    '''客户信用风险模型'''
    def __init__(self, config):
        self.factor = 20 / np.log(2)
        self.offset = 600 - self.factor * np.log(0.1)
        self.dim_sql = "SELECT a.*, b.parent_yearly_avg_sale_qty, b.parent_yearly_avg_sale_amt, b.parent_before_3month_actual_sale_qty, b.parent_sale_qty_yoy, b.yearly_overdue15_count, b.yearly_overdue15_rate, b.three_month_collection_ratio  FROM GF_DWR_MODEL.XYG_GFOEN_DWR_CCR_CUSTOMER_RISKS a LEFT JOIN GF_DWR_MODEL.XYG_GFOEN_DWR_CCR_CUSTOMER_INDICATORS_ADD b  on a.customer_name = b.customer_name and a.customer_number = b.customer_number and a.year_month = b.year_month where a.year_month>=TO_CHAR(CURRENT_TIMESTAMP - INTERVAL '3 MONTH', 'YYYY-MM')  and a.year_month<= TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM')" 
        self.del_sql = "DELETE FROM GF_DWR_MODEL.XYG_GFOEN_DWR_CCR_CUSTOMER_SCORES WHERE year_month=TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM') "

        # self.dim_sql = "SELECT * FROM GF_DWR_MODEL.XYG_GFOEN_DWR_CCR_CUSTOMER_RISKS WHERE YEAR_MONTH = TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM') ORDER BY CUSTOMER_NAME"
        self.key = config['model']['key']
        self.iv = config['model']['iv']
        self.host = config['model']['host']
        self.database = config['model']['database']
        self.user = config['model']['user']
        self.password = config['model']['password']
        
    # 解密过程逆着加密过程写
    def AES_de(self, data, key, iv):
        # 将密文字符串重新编码成二进制形式
        data = data.encode("utf-8")
        # 将base64的编码解开
        data = base64.b64decode(data)
        # 创建解密对象
        AES_de_obj = AES.new(self.key.encode("utf-8"), AES.MODE_CBC, self.iv.encode("utf-8"))
        # 完成解密
        AES_de_str = AES_de_obj.decrypt(data)
        # 去掉补上的空格
        AES_de_str = AES_de_str.strip()
        # 对明文解码
        AES_de_str = AES_de_str.decode("utf-8")
        return AES_de_str
    
    def dwsConnect(self, host, database, user, password):# 连接数据库
        '''通过psycopg2包连接数据库，接入模型所用中间表'''
        con = None
        # 创建psycopg2连接，输入连接信息
        try:
            con = psycopg2.connect(
                host = host,
                database = database,
                user = user,
                password = password,
                port = 8000
            )
            con.set_client_encoding('utf8') #设置client编码格式为utf8来解析中文
            cur = con.cursor()
            cur.execute(self.del_sql) #删除当月数据
            con.commit()
            cur.execute(self.dim_sql) #cursor传递sql语句
            version = cur.fetchall() #cursor执行sql语句并返回查询结果
    
        except psycopg2.DatabaseError as e: #打印错误
            print(e)
            sys.exit(1)
    
        finally: #关闭连接
            if con:
                con.close()
        df = pd.DataFrame(version)
        print('数据表传入完成')
        return df, df.shape[0]

        
    def rename(self, df):
        '''表头重命名'''
        df.columns = ['客户名称', 
              '客户编码', 
              '年月', 
              '现金比率',
              '资产负债率',
              '应收账款周转天数',
              '总资产周转率',
              '币种代码',
              '近一年月均销售量',
              '近一年月均销售额',
              '近一年销售量增长率',
              '合作总时长',
              '上次交易时间',
              '信义占比',
              '营业收入同比增长率',
              '销售毛利率',
              '毛利同比增长率',
              '销售净利率',
              '净利同比增长率',
              '客户产能产量',
              '实缴资本',
              '企业性质',
              '成立时间',
              '近一年法人变更次数',
              '近一年大股东变更次数',
              '近一年营业范围变更次数',
              '历史诉讼记录',
              '历史行政处罚次数',
              '关联企业高风险次数',
              '董高监失信人被执行次数',
              '是否进入政务的企业黑名单',
              '近一年逾期比例',
              '近一年逾期次数',
              '是否是问题客户',
                '客户集团过去一年月均销量',
                '客户集团过去一年月均销售额',
                '客户集团过去3个月实际销量',
                '客户集团销量同比',
                '客户过去一年逾期超15天次数',
                  '逾期15比例',  #当前
                  '近3月回款比例'
                 ]
        print('数据表重命名表头完成')
        return df
        
    def dataCleaning(self, df):
        df = df[[col for col in df.columns if col != '是否是问题客户'] + ['是否是问题客户']]
        df_ym = df['年月']
        # 删除不进入模型的字段
        df.drop(columns=['客户名称','客户编码', '年月', '币种代码'], inplace=True)
        
        #识别出空的位置，在后续计算分数时，空值的位置不参与计算
        df['实缴资本']  = df['实缴资本'].replace('-',np.nan)
        df['客户产能产量']  = df['客户产能产量'].replace(0,np.nan)
        null_positions = df.drop(['是否是问题客户'],axis=1).isnull()
        matrix_01 = null_positions.astype(int).applymap(lambda x: 0 if x else 1)
        matrix_01_new = matrix_01.replace(0,np.nan)
        
        # 二分型离散变量编码处理
        df['是否是问题客户'] = df['是否是问题客户'].apply(lambda x: 1 if x == '是' else 0)
        df['是否进入政务的企业黑名单'] = df.apply(lambda x: 1 if str(x['是否进入政务的企业黑名单']) == '是' else 0, axis = 1)
        df['企业性质'] = df['企业性质'].fillna('-9999')
        df['企业性质'] = df.apply(lambda x: 1 if '有限责任公司' in x['企业性质'] else 0, axis = 1)
        df['企业性质'] = df['企业性质'].apply(pd.to_numeric)
        
        # 月销量没有数据的，计为0
        df['近一年月均销售量'] = df['近一年月均销售量'].fillna(0)
        df['近一年月均销售额'] = df['近一年月均销售额'].fillna(0)
    
        # 上次交易时间从时间戳转化为距今的天数
        df['上次交易时间'] = df['上次交易时间'].apply(lambda x: int(str((pd.Timestamp.now() - x)).split(' ')[0]))
        # 实缴资本美元转化为人民币，剔除中文和不需要的字符
    #     df['实缴资本'] = df[df['实缴资本'].isna() == False].apply(lambda x: x['实缴资本'].replace('-', '-9999'), axis = 1)
        df['实缴资本'] = df[df['实缴资本'].isna() == False].apply(lambda x: '0' if x['实缴资本'] == '' else x['实缴资本'], axis = 1)
        df['实缴资本'] = df[df['实缴资本'].isna() == False].apply(lambda x: str(float(x['实缴资本'].split('万')[0]) * 7) + '万人民币' if x['实缴资本'] != '' and x['实缴资本'] != '0' and '美元' in x['实缴资本'] else x['实缴资本'], axis = 1)
        df['实缴资本'] = df[df['实缴资本'].isna() == False].apply(lambda x: x['实缴资本'].replace('万人民币', '').replace('万美元', '').replace('万', ''), axis = 1)
    #     df['现金比率'].fillna(-9999, inplace=True) 
    #     df['资产负债率'].fillna(-9999, inplace=True)
    #     df['应收账款周转天数'].fillna(-9999, inplace=True) 
    #     df['总资产周转率'].fillna(-9999, inplace=True) 
        print('数据清洗完成')
        return df,df_ym,matrix_01,matrix_01_new
        
    def knn(self, df):
        '''使用knn算法进行缺失值填补'''
        df_columns = list(df.columns) # 提取df里所有列
        # 删除不参与knn近邻的列，即空值列
        removed_columns = ['信义占比', '客户产能产量', '营业收入同比增长率', '销售毛利率', '毛利同比增长率', '销售净利率', '净利同比增长率', '现金比率', '资产负债率', '应收账款周转天数', '总资产周转率']
        for i in removed_columns:
            df_columns.remove(i)
        # 将object类型转化为numeric
        for i in df_columns:
            df[i] = df[i].apply(pd.to_numeric)
        # knn预测
        imputer = KNNImputer(n_neighbors=25) #n_neighbors=25保证有区分度
        df_knn = imputer.fit_transform(df[df_columns])
        df_knn = pd.DataFrame(df_knn, columns = df_columns)
        # 将knn预测的值替换到原始df中
        columns = ['近一年月均销售量', '近一年月均销售额', '近一年销售量增长率', '实缴资本']
        for i in columns:
            df[i] = np.array(df_knn[i])
        # 将其他含有空值的列用0填补
        df.fillna(0, inplace=True)
        print('knn缺失值填补完成')
        return df
        
    def SMOTE_(self, df):
        '''使用SMOTE对坏样本进行过采样'''
        # 将df拆解为目标变量与特征变量
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1:]
        # 生成SMOTE工具，拟合x和y
        sm = SMOTE(sampling_strategy= 1, random_state=0)
        X_res, y_res = sm.fit_resample(x, y)
        # 将生成的新坏样本数据与原样本数据合并
        df = pd.concat([X_res, y_res], axis = 1)
        # 计算目前的好坏样本比值，后续在计算woe值时使用
        rate = df['是否是问题客户'].sum() / (df['是否是问题客户'].count() - df['是否是问题客户'].sum())
        print('SMOTE采样完成')
        return df, rate
    
    def get_mae(self, max_leaf_nodes, train_X, val_X, train_y, val_y):
        '''计算mae'''
        # best_leaf_nodes 根据mae调参
        model= DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state=0) #初始化模型
        model.fit(train_X, train_y)
        val_prediction_y= model.predict(val_X)
        mae= mean_absolute_error(val_y, val_prediction_y) #计算此模型的mae值
        return mae
    
    def decisionTree(self, df):
        '''得出最佳叶节点数量'''
        # 训练集-测试集拆分
        x = df.iloc[:,:-1]
        y = df.iloc[:,-1::]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        # 计算各best_leaf_node的mae，将不再下降的mae节点设置为最佳节点
        my_mae_prev = 0
        for leaf_nodes in [i for i in range(2,100)]: #叶节点从2遍历到50
            my_mae= self.get_mae(leaf_nodes, x_train, x_test, y_train, y_test)
            if my_mae_prev == my_mae: #当mae值不再下降，取此时的叶节点数目为最佳参数
                best_leaf_nodes = leaf_nodes
                break
            my_mae_prev = my_mae
        print('最佳叶节点数量计算完成')
        return x, y, best_leaf_nodes
    
    def get_woe_data(self, df, cut, rate):
        '''计算woe值'''
        grouped = df["是否是问题客户"].groupby(cut,as_index = True).value_counts()
        print(grouped)
        woe = np.log(grouped.unstack().iloc[:,1]/grouped.unstack().iloc[:,0]/rate) #使用分享内odd比值除以样本总体odd比值，odd比值为坏客户和好客户的比例
        return woe
    
    def optimal_binning_boundary(self, df, variable, best_leaf_nodes, x, y, rate):
        '''利用决策树获得最优分箱的边界值列表'''
        boundary = []  # 待return的分箱边界值列表

        x = x.values
        y = y.values

        clf = DecisionTreeClassifier(criterion='entropy',    #“信息熵”最小化准则划分
                                     max_leaf_nodes= best_leaf_nodes,       # 最大叶子节点数
                                     min_samples_leaf=0.05, # 叶子节点最小样本比例
                                     random_state = 0)  # 叶子节点样本数量最小占比

        clf.fit(x.reshape(-1, 1), y)  # 训练决策树

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold

        for i in range(n_nodes):
            if children_left[i] != children_right[i]:  # 获得决策树节点上的划分边界值
                boundary.append(threshold[i])
        boundary.sort()

        min_x = x.min() - 0.1
        max_x = x.max() + 0.1  # +0.1是为了考虑后续groupby操作时，能包含特征最大值的样本
        boundary = [min_x] + boundary + [max_x]

        # 算出此种方法分箱后的woe值
        cut = pd.cut(df[variable], boundary, labels=False)
        cut_woe = self.get_woe_data(df, cut, rate)

        # 如果某个分箱内全部为好客户或坏客户，删除cut值并做分箱合并
        for i in range(len(cut_woe)):
            if i == 0 and np.isnan(cut_woe[i]): #如果第一个分箱为空值，删除第二个cut值，合并一二箱
                boundary[i+1] = 'delete'
            elif i == len(cut_woe) - 1 and np.isnan(cut_woe[i]): #如果最后一个分箱为空值，删除倒数第二个cut值，合并倒数一二箱
                boundary[i] = 'delete'
            elif np.isnan(cut_woe[i]): #如果中间某个分箱出现空值，删除前一个cut值
                boundary[i] = 'delete'
                
        # 删除delete项，即需要被删除的项
        while 'delete' in boundary:
            boundary.remove('delete')
        
        # 返回决策树分箱后的边界
        return boundary
    
    def boundaries(self, df, x, y, best_leaf_nodes, rate):
        '''计算全部变量的分箱边界'''
        boundaries = []
        for i in x.columns:
            boundary = self.optimal_binning_boundary(df, i, best_leaf_nodes, x[i], y['是否是问题客户'], rate)
            boundaries.append(boundary)
        print('最佳分箱边界计算完成')
        return boundaries
    
    def get_woe(self, df, x, boundaries, rate):
        '''根据分箱边界，得出每行数据被分配到了哪个箱中，和这个分箱的woe值'''
        cuts = []
        cut_woes = []
        for i in range(len(x.columns)):
            cut = pd.cut(x[x.columns[i]], boundaries[i], labels=False)
            cut_woe = self.get_woe_data(df, cut, rate)
            cuts.append(cut)
            cut_woes.append(cut_woe)
        print('woe计算完成')
        return cuts, cut_woes #返回所有变量的分箱分配情况和woe值
        
    def get_IV_data(self, df, cut, cut_woe):
        '''计算IV的公式'''
        grouped = df['是否是问题客户'].groupby(cut, as_index = True).value_counts()
        cut_IV = ((grouped.unstack().iloc[:,1] / df['是否是问题客户'].sum() - grouped.unstack().iloc[:,0] / (df['是否是问题客户'].count() - df['是否是问题客户'].sum())) * cut_woe).sum()   
        return cut_IV
    
    def get_IV(self, df, x, cuts, cut_woes):
        '''遍历所有变量，计算出每个变量的IV值，作为变量选择的依据'''
        cut_IVs = []
        for i in range(len(x.columns)):
            cut_IV = self.get_IV_data(df, cuts[i], cut_woes[i])
            cut_IVs.append(cut_IV)
        print('IV值计算完成')
        return cut_IVs
    
    def replace_data(self, cut,cut_woe):
        '''替换原始数据表数据的函数'''
        a = []
        for i in cut.unique():
            a.append(i)
            a.sort()
        for m in range(len(a)):
            cut.replace(a[m], cut_woe.values[m], inplace=True)
        return cut
    
    def replace_IV(self, df, x, cut_IVs, cuts, cut_woes):
        '''将woe值替换原始数据表数据的实施'''
        IV = pd.DataFrame()
        IV['变量名称'] = x.columns
        IV['iv'] = cut_IVs
        # IV = IV[IV['iv'] >= 0.02] #筛选IV值大于0.02的变量，作为模型的输入
        
        df_woe = pd.DataFrame()
        for i in IV.index:
            df_woe[IV['变量名称'][i]] = self.replace_data(cuts[i], cut_woes[i])
        df_woe["是否是问题客户"] = df["是否是问题客户"]
        print('IV替换完成')
        return df_woe
    
    def logisticRegression(self, df_woe):
        '''逻辑回归模型训练，并返回各变量权重/系数'''
        # 训练集-预测集划分
        x = df_woe.iloc[:,:-1]
        y = df_woe.iloc[:,-1::]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
        # 模型训练
        model = LogisticRegression(random_state=0)
        clf = model.fit(x_train, y_train)
        coe=clf.coef_ #取模型的系数
        print('逻辑回归模型运行完成')
        return coe
    
    def get_score(self, coe, woe, factor):
        '''各分箱的分值计算公式'''
        score = []
        for i in woe:
            scoring = round(-i * coe * factor, 4)
            score.append(scoring)
        return score
    
    def scores(self, x, coe, cut_woes):
        '''循环所有变量，计算出各分箱对应的分数'''
        scores = []
        for i in range(len(x.columns)):
            score = self.get_score(coe[0][i], cut_woes[i], self.factor)
            scores.append(score)
        print('各分箱分值计算完成')
        return scores
    
    #.各变量对应的分数求和，算出每个用户的总分
    def compute_score(self, series,bins,score):
        '''为原始数据表的各变量的各分箱替换分值'''
        list_ = []
        list_level = []
        i = 0
        while i < len(series):
            value = series[i]
            j = len(bins) - 2
            m = len(bins) - 2
            while j >= 0:
                if value >= bins[j]:
                    j = -1
                else:
                    j -= 1
                    m -= 1
            list_.append(score[m])
            list_level.append(m)
            i += 1
        return list_,list_level
    
    # 数据量有限的，转换分数分差不宜过大
    def trans_A(self, df, matrix_01, attribute,range0 = (20,98)):
        aa = df[attribute] * matrix_01[attribute]
        n = len(aa.unique())
        if n<50:
            range1 = (85 - round(n/4,0), 85 + round(n/4,0))
        else :
            range1 = range0
        scalar = MinMaxScaler(feature_range=range1)
        
        scores = scalar.fit_transform(aa.values.reshape(-1, 1))
        return scores
        
    def score_convert(self, df, df_original, length, scores, boundaries,df_ym,matrix_01):
        '''
        1. 替换分值
        2. 客户总评分与雷达图5个维度评分的计算
        3. 客户评级的判断
        4. 增加年月字段
        5. 根据词根词条，重命名表头
        '''
        # 将范围限定到初始的客户群中，舍弃SMOTE创建的客户
        df = df.iloc[:length, :]
        list_levels = []
        # 计算雷达图分数  # nan值 分数0
        for i in range(len(df.columns) - 1):
            df['x'+str(i+1)],list_level= pd.Series(self.compute_score(df[df.columns[i]],boundaries[i], scores[i]))
            list_levels.append(list_level)
    
        # 销售量和销售额方面进行人工修正，销售越多，分值越高
        # df['x5'] = df['x5'] + 10*np.array(list_levels[4] ) #近一年月均销售量
        # df['x6'] = df['x6'] + 10*np.array(list_levels[5] ) #近一年月均销售额
        # df['x7'] = df['x7'] + 10*np.array(list_levels[6] ) #近一年销售量增长率
        df['x9'] = df['x9'] + 5*len(scores[i]) - 5*np.array(list_levels[8] ) #近一年月均销售量
        df['x29'] = df['x29'] + 10*np.array(list_levels[28] ) #近一年月均销售量
        df['x30'] = df['x30'] + 10*np.array(list_levels[29] ) #近一年月均销售额
        df['x31'] = df['x31'] + 10*np.array(list_levels[30] ) #近一年销售量增长率
    
        # 对于某些和问题客户强相关的字段，从业务角度修正分数
        df['x24'] = df.apply(lambda x: x['x24'] - x['历史行政处罚次数'] * 5, axis=1)
        df['x26'] = df.apply(lambda x: x['x26'] - x['董高监失信人被执行次数'] * 100, axis=1)
        df['x27'] = df.apply(lambda x: x['x27'] - x['是否进入政务的企业黑名单'] * 100, axis=1)
        # df['x29'] = df.apply(lambda x: x['x29'] - x['近一年逾期次数'] * 100, axis=1)
    
        #交易金额小的，减分。
        df['enterprise_solvency'] = df['x1'] + \
            df['x2'] + df['x3'] + df['x4']  # 偿债能力  
        # df['cooperation_scale'] = df['x5'] + df['x6'] + \
        #     df['x7'] + df['x8'] + df['x9'] + df['x10'] + \
        #     df['x30'] + df['x31'] + df['x32'] + df['x33']# 合作规模  
        df['cooperation_scale'] =     df['x8'] + df['x9'] + df['x10'] + \
                    df['x30'] + df['x31'] + df['x32'] + df['x33']# 合作规模  
        df['business_performance'] = df['x11'] + df['x12'] + \
            df['x13'] + df['x14'] + df['x15'] + df['x16']  # 经营效益 
    
        df['enterprise_quality'] = df['x17'] + df['x18'] + df['x19'] + df['x20'] + \
            df['x21'] + df['x22'] + df['x23'] + df['x24'] + df['x25']  # 企业素质  
    
        df['agreement_performance'] = df['x26'] + \
            df['x27'] + df['x28'] + df['x29'] + \
            df['x34'] + df['x35'] + df['x36'] # 履约情况 
        
        # mtrix_01 中每个板块做计算，若为0，则评分结果应该为空
        matrix_01['enterprise_solvency'] = matrix_01.iloc[:,:4].sum(axis=1).apply(lambda x: np.nan if x==0  else 1)
        matrix_01['cooperation_scale'] = matrix_01.iloc[:, list(range(4, 10))+list(range(29, 33))].sum(axis=1).apply(lambda x: np.nan if x==0  else 1)
        matrix_01['cooperation_scale'] = matrix_01.iloc[:, list(range(7, 10))+list(range(29, 33))].sum(axis=1).apply(lambda x: np.nan if x==0  else 1)
        matrix_01['business_performance'] = matrix_01.iloc[:,10:16].sum(axis=1).apply(lambda x: np.nan if x==0  else 1)
        matrix_01['enterprise_quality'] = matrix_01.iloc[:,16:25].sum(axis=1).apply(lambda x: np.nan if x==0  else 1)
        matrix_01['agreement_performance'] = matrix_01.iloc[:, list(range(25, 29))+list(range(33, 36))].sum(axis=1).apply(lambda x: np.nan if x==0  else 1)
        
        # range0 = (20,98)
        # for attribute in ('enterprise_solvency', 'cooperation_scale', 'business_performance', 'enterprise_quality', 'agreement_performance'):
        #     df[attribute] = self.trans_A(df, matrix_01, attribute, range0)
        
        # weights1 = matrix_01['enterprise_solvency'].apply(
        #     lambda x: 0 if np.isnan(x) else 1) * 0.1
        # weights2 = matrix_01['cooperation_scale'].apply(
        #     lambda x: 0 if np.isnan(x) else 1) * 0.3
        # weights3 = matrix_01['business_performance'].apply(
        #     lambda x: 0 if np.isnan(x) else 1) * 0.1
        # weights4 = matrix_01['enterprise_quality'].apply(
        #     lambda x: 0 if np.isnan(x) else 1) * 0.2
        # weights5 = matrix_01['agreement_performance'].apply(
        #     lambda x: 0 if np.isnan(x) else 1) * 0.3
        # df['customer_score'] = (df['enterprise_solvency'] * weights1 + df['cooperation_scale'] * weights2 + df['business_performance'] * weights3 + df['enterprise_quality'] * weights4 + df['agreement_performance'] * weights5)/(weights1+weights2+weights3+weights4+weights5)
        # # scalar = MinMaxScaler(feature_range=range0) 
        # # df['customer_score'] = scalar.fit_transform(df['customer_score'].values.reshape(-1, 1))
        
        # # 计算超越x%客户字段
        # percentile=[]
        # scores=sorted(list(df['customer_score']))
        # for i in range(len(df)):
        #     percentile.append(
        #         '超越' + str(round(scores.index(df['customer_score'][i])/length * 100, 0)) + '%客户')
        # df['customer_performance']=percentile
    
        # for x in ('customer_score', 'enterprise_solvency', 'cooperation_scale', 'business_performance', 'enterprise_quality', 'agreement_performance'):
        #     df[x] = round(df[x], 0)
    
        # # 得出得分评级
        # df['customer_rate'] = df['customer_score'].apply(lambda x: '优质客户' if x >= 85 else '良好客户' if x >= 60 else '一般客户')
        # df['customer_rate'] = df.apply(lambda x: '问题客户' if x['是否是问题客户'] == 1 else x['customer_rate'], axis=1)
        # # 添加日期字段
        # df['year_month'] = df_ym  #datetime.datetime.now().strftime('%Y-%m')
        range0 = (20,98)  # 单项分数20,98，最终分数30,98
        for attribute in ('enterprise_solvency', 'cooperation_scale', 'business_performance', 'enterprise_quality', 'agreement_performance'):
            df[attribute] = self.trans_A(df, matrix_01, attribute, range0)
        
        att0 = np.nan_to_num(df['enterprise_quality'])
        att1 = np.nan_to_num(df['cooperation_scale']) 
        att2 = np.nan_to_num(df['business_performance']) 
        att3 = np.nan_to_num(df['enterprise_quality']) 
        att4 = np.nan_to_num(df['agreement_performance']) 
        ww_00 = np.nan_to_num(matrix_01['enterprise_quality'])
        ww_01 = np.nan_to_num(matrix_01['cooperation_scale'])
        ww_02 = np.nan_to_num(matrix_01['business_performance'])
        ww_03 = np.nan_to_num(matrix_01['enterprise_quality'])
        ww_04 = np.nan_to_num(matrix_01['agreement_performance'])
        weights = [0.05,0.4,0.05,0.2,0.3]   # 每项分数的权重
        df['customer_score'] = (att0* weights[0] + att1 * weights[1] + att2 * weights[2] + att3 * weights[3] + att4 * weights[4])/(weights[0]*ww_00 +weights[1]*ww_01+weights[2]*ww_02+weights[3]*ww_03+weights[4]*ww_04)
        scalar = MinMaxScaler(feature_range=(30,98)) 
        df['customer_score'] = scalar.fit_transform(df['customer_score'].values.reshape(-1, 1))  
        
        # 计算超越x%客户字段
        percentile=[]
        scores=sorted(list(df['customer_score']))
        for i in range(len(df)):
            percentile.append(
                '超越' + str(round(scores.index(df['customer_score'][i])/length * 100, 1)) + '%客户')
        df['customer_performance']=percentile
        
        for x in ('customer_score', 'enterprise_solvency', 'cooperation_scale', 'business_performance', 'enterprise_quality', 'agreement_performance'):
            df[x] = round(df[x], 0)
        
        # 得出得分评级
        df['customer_rate'] = df['customer_score'].apply(lambda x: '优质客户' if x >= 85 else '良好客户' if x >= 60 else '一般客户')
        df['customer_rate'] = df.apply(lambda x: '问题客户' if x['是否是问题客户'] == 1 else x['customer_rate'], axis=1)
        # 添加日期字段
        df['year_month'] = df_ym  #datetime.datetime.now().strftime('%Y-%m')
        # 重命名客户名称字段
        df_original.rename(columns={'客户名称':'customer_name', '客户编码':'customer_number'},inplace=True)
        df_original = df_original[['customer_name', 'customer_number']]
        df = df[['year_month', 'customer_score', 'enterprise_quality', 'agreement_performance', 'enterprise_solvency', 'cooperation_scale', 'business_performance', 'customer_rate', 'customer_performance']]
        df_output = pd.concat([df_original, df], axis=1)
        df_output = df_output[df_output['year_month']==datetime.datetime.now().strftime('%Y-%m')]
        print('结果表输出完成')
        return df_output

    # 数据回写入dws数据库表内
    def dataRewrite(self, df_output, table_name, host, database, user, password):
        '''结果表回写到dws数据库中'''
        # 连接数据库
        password = parse.quote_plus(password)
        SQLALCHEMY_DATABASE_URI = 'postgresql://' + user + ':'  + password + '@' + host + ':8000' + '/' + database
        engine = create_engine(SQLALCHEMY_DATABASE_URI, client_encoding='utf8')
        # 写入数据库
        df_output.to_sql(schema='gf_dwr_model', con=engine, name = table_name, if_exists='append', index=False)
        
    def apply(self):
        '''调度全部脚本'''
        ###获取配置文件里的敏感信息，进行解密
        host = self.AES_de(self.host, self.key, self.iv)
        database = self.AES_de(self.database, self.key, self.iv)
        user = self.AES_de(self.user, self.key, self.iv)
        password = self.AES_de(self.password, self.key, self.iv)
        
        ##获取数据表
        df, length = self.dwsConnect(host, database, user, password)
        df = df.drop([34,35],axis=1)  # 去掉etl时间，更新时间

        ###重命名表头
        df = self.rename(df)
        df_original = df.copy()
        
        ##数据清洗，独热编码
        df,df_ym,matrix_01,matrix_01_new = self.dataCleaning(df)
        
        df = df.replace(np.nan, '-9999')   #先统一处理为-9999，如果指标绝大多数为空值，最后需要手动调整权重系数
        
        ##SMOTE过采样
        df, rate = self.SMOTE_(df)
        
        ##决策树分箱
        x, y, best_leaf_nodes = self.decisionTree(df)
        
        ###计算分箱边界
        boundaries = self.boundaries(df, x, y, best_leaf_nodes, rate)
        
        ##得出分箱分配结果及对应woe值
        cuts, cut_woes =  self.get_woe(df, x, boundaries, rate)

        ###计算每个变量的IV值
        cut_IVs =  self.get_IV(df, x, cuts, cut_woes)
        list_name = df.columns
        weights = dict()
        for i ,tt in enumerate(list_name[:-1]):
            weights[tt] = cut_IVs[i]
        
        ###将woe值替换原数据表数据，woe编码完成
        df_woe =  self.replace_IV(df, x, cut_IVs, cuts, cut_woes)
        
        ###训练逻辑回归模型，得到各变量系数/权重
        coe =  self.logisticRegression(df_woe)
        
        ###计算各分箱得分
        scores =  self.scores(x, coe, cut_woes)
        
        ###汇总得分，得出客户总得分及雷达图5个维度分别的得分
        df_output =  self.score_convert(df, df_original, length, scores, boundaries,df_ym, matrix_01)
        
        ##将结果回写入dws
        self.dataRewrite(df_output, 'xyg_gfoen_dwr_ccr_customer_scores', host, database, user, password)
        print(len(df_output))
        print('数据回写dws完成')
        
#实例化与脚本执行
if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("/home/qc/iiot_config/conf.ini")
    ccr = CustomerCreditRisk(config)
    ccr.apply()
    print('Success')
    