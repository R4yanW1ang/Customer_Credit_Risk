## PYTHON 
## ******************************************************************** ##
## function: 天眼查入湖
## 来源表：GF_COM.XYG_GFOEN_COM_CFG_MODEL_CUSTOMER_LIST
## 目标表: ods.tianyancha_basic_info
## ods.tianyancha_risk
## author: hw_wangzirui
## create time: 2022/07/25 11:19:10 GMT+08:00
## ******************************************************************** ##
#encoding:utf-8
import psycopg2
import pandas as pd
import numpy as np
import time
import datetime
import os
import json
import requests
import time
import importlib, sys
from sqlalchemy import create_engine
from urllib import parse
import base64
from Crypto.Cipher import AES
import configparser

class Tianyancha():
    """天眼查入湖"""
    def __init__(self, config):
        self.headers = {'Authorization': "ce3f38f8-d269-4d5d-9f1e-18004f199c58"}
        self.dim_sql = "SELECT * FROM GF_COM.XYG_GFOEN_COM_CFG_MODEL_CUSTOMER_LIST WHERE YEAR_MONTH = TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM')"
        self.columns_customers = ['customer_name', 'year_month', 'actual_capital', 'company_org_type', 'establish_date', 'staff_number_range', 'reg_status', 'legal_person_name', 'finance_black_list', 'high_risk_count']
        self.columns_dates = ['customer_name', 'year_month', 'risk_type', 'risk_detail']
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
            cur.execute(self.dim_sql) #cursor传递sql语句
            version = cur.fetchall() #cursor执行sql语句并返回查询结果

        except psycopg2.DatabaseError as e: #打印错误
            print('Error:' + str(e))
            sys.exit(1)

        finally: #关闭连接
            if con:
                con.close()

        # 将返回的数据创建为df，创建一个列表来存储所有客户名称
        df = pd.DataFrame(version, columns=['客户名称', '客户编码', '年月'])
        customer_list = list(df['客户名称'])

        return customer_list
    
    
    # 获取客户基本信息字段
    def basicInfo(self, response):
        if json.loads(response.text)['result'] == None: # 如果接口返回数据为空，则返回空值
            return [None, None, None, None, None, None]
        content = json.loads(response.text)['result'] # 解析json
        # 开始提取字段信息
        actualCapital = content['actualCapital'] # 实缴资本
        companyOrgType = content['companyOrgType'] # 企业性质
        establishTime = content['estiblishTime']
        if establishTime != None:
            establishTime = time.strftime("%Y-%m-%d", time.localtime(int(establishTime) / 1000)).split('-') # 成立时间
            curr_time = time.time()
            curr_time = time.strftime("%Y-%m", time.localtime(curr_time)).split('-')
            time_difference = (int(curr_time[0]) - int(establishTime[0])) * 12 + (int(curr_time[1]) - int(establishTime[1]))
        else:
            time_difference = 0
        staffNumRange = content['staffNumRange'] # 人员规模
        regStatus = content['regStatus'] # 经营状态
        legalPersonName = content['legalPersonName'] # 企业法人
        print(actualCapital, companyOrgType, time_difference, staffNumRange, regStatus, legalPersonName)
        return [actualCapital, companyOrgType, time_difference, staffNumRange, regStatus, legalPersonName]
    
    # 涉金融政务黑名单 & 关联企业高风险次数 字段获取
    def enterpriseRisk(self, response):
        if json.loads(response.text)['result'] == None:
            return [None, None]
        black_list = '否'
        high_risk = 0
        for i in json.loads(response.text)['result']['riskList'][0]['list']:
            if i['title'] == '涉金融黑名单':
                black_list = '是'
            if i['tag'] == '高风险':
                high_risk += i['total']
        return [black_list, high_risk]
    
    
    # 变更事项 字段获取
    def changeItem(self, response):
        if json.loads(response.text)['result'] == None:
            return [[None, None, None]]
        content = json.loads(response.text)['result']['items'] # 解析json
        changes = []
        for i in content:
            changeTime = i['changeTime'] # 变更时间
            if changeTime == '':
                changeTime = (datetime.datetime.now() + datetime.timedelta(days=-30)).strftime('%Y-%m-%d')
            types = i['changeItem'] # 变更项
            content = '“' + i['contentBefore'] + '”' + '变更为' + '“' + i['contentAfter'] + '”' # 变更内容
            changes.append([changeTime, types, content])
        return changes
    
    
    # 行政处罚次数 字段获取
    def adminPunishment(self, response):
        if json.loads(response.text)['result'] == None:
            return [[None, None, None]]
        content = json.loads(response.text)['result']['items'] # 解析json
        punish = []
        for i in content:
            decisionDate = i['decisionDate'] # 处罚日期
            types = '行政处罚' # 风险类型
            content = i['reason'] + ' ' + i['content'] # 处罚结果/内容
            punish.append([decisionDate, types, content])
        return punish
    
    # 限制消费令次数 字段获取
    def sumptuaryOrder(self, response):
        if json.loads(response.text)['result'] == None:
            return [[None, None, None]]
        content = json.loads(response.text)['result']['items'] # 解析json
        sumptuary = []
        for i in content:
            caseCreateTime = i['caseCreateTime'] # 立案时间
            caseCreateTime = time.strftime("%Y-%m-%d", time.localtime(int(caseCreateTime) / 1000))
            types = '限制消费令' # 风险类型
            content = '该公司的' + i['xname'] + '被法院判定为限制消费' # 处罚结果/内容
            sumptuary.append([caseCreateTime, types, content])
        return sumptuary
    
    # 失信被执行人次数 字段获取
    def dishonestPerson(self, response):
        if json.loads(response.text)['result'] == None:
            return [[None, None, None]]
        content = json.loads(response.text)['result']['items'] # 解析json
        dishonest = []
        for i in content:
            regdate = i['regdate'] # 立案时间
            regdate = time.strftime("%Y-%m-%d", time.localtime(int(regdate) / 1000))
            types = '失信被执行人' # 风险类型
            content = '该公司' + i['disrupttypename'] # 失信被执行人行为具体情形
            dishonest.append([regdate, types, content])
        return dishonest
    
    # 法律诉讼次数 字段获取
    def historyLawsuit(self, response):
        if json.loads(response.text)['result'] == None:
            return [[None, None, None]]    
        content = json.loads(response.text)['result']['items'] # 解析json
        lawsuit = []
        for i in content:
            judgeTime = i['judgeTime'] # 裁判日期
            if judgeTime == '':
                continue
            if '00000' in judgeTime:
                judgeTime = time.strftime("%Y-%m-%d", time.localtime(int(judgeTime) / 1000))
            types = '法律诉讼' # 风险类型
            content = '该公司涉及' + i['caseReason'] # 案由
            lawsuit.append([judgeTime, types, content])
        return lawsuit
    
    # 调取客户基本信息函数，循环插入客户名称，整合返回数据至df_output
    def customerBased(self, customer_list):
        df_output = pd.DataFrame()
        for customer in customer_list:
            response_bi = requests.get('http://open.api.tianyancha.com/services/open/ic/baseinfo/normal?keyword='+customer, headers=self.headers)
            response_er = requests.get('http://open.api.tianyancha.com/services/open/risk/riskInfo/2.0?&keyword='+customer, headers=self.headers)
            year_month = (datetime.datetime.now()).strftime('%Y-%m')
            result_bi = self.basicInfo(response_bi)
            result_er = self.enterpriseRisk(response_er)
            result = result_bi + result_er
            df_ongoing = pd.DataFrame([np.array([customer] + [year_month] + result)])
            print(df_ongoing)
            df_output = pd.concat([df_output, df_ongoing], axis = 0)
        df_output.columns = self.columns_customers
        return df_output
    
    # 调取客户风险信息的数个函数，循环插入客户名称，整合返回数据至df_output
    def dateBased(self, customer_list):
        df_output = pd.DataFrame()
        for customer in customer_list:
            response_ci = requests.get('http://open.api.tianyancha.com/services/open/ic/changeinfo/2.0?keyword='+customer, headers=self.headers)
            response_ap = requests.get('http://open.api.tianyancha.com/services/open/mr/punishmentInfo/3.0?keyword='+customer, headers=self.headers)
            response_so = requests.get('http://open.api.tianyancha.com/services/open/jr/consumptionRestriction/2.0?keyword='+customer, headers=self.headers)
            response_dp = requests.get('http://open.api.tianyancha.com/services/open/jr/dishonest/2.0?&keyword='+customer, headers=self.headers)
            response_hl = requests.get('http://open.api.tianyancha.com/services/open/jr/lawSuit/3.0?keyword='+customer, headers=self.headers)
            result_ci = self.changeItem(response_ci)
            result_ap = self.adminPunishment(response_ap)
            result_so = self.sumptuaryOrder(response_so)
            result_dp = self.dishonestPerson(response_dp)
            result_hl = self.historyLawsuit(response_hl)
            for i in [result_ci, result_ap, result_so, result_dp, result_hl]:
                for j in range(len(i)):
                    i[j] = [customer] + i[j]
                df_ongoing = pd.DataFrame(i)
                df_output = pd.concat([df_output, df_ongoing], axis = 0)
        df_output.columns = self.columns_dates
        df_output = df_output.dropna() # 删除空值

        # 营业范围 字段替换
        df_output['risk_type'] = df_output['risk_type'].apply(lambda x: '营业范围变更' if '范围' in x else x)
        # 股东变更 字段替换
        df_output['risk_detail'] = df_output.apply(lambda x: '股东发生变更' if x['risk_type'] == '法人股东' or x['risk_type'] == '自然人股东' else x['risk_detail'], axis=1)
        df_output['risk_type'] = df_output['risk_type'].apply(lambda x: '股东变更' if x == '法人股东' or x == '自然人股东' else x)
        # 法定代表人变更 字段替换
        df_output['risk_type'] = df_output['risk_type'].apply(lambda x: '法定代表人变更' if x == '法定代表人（负责人）变更' or x == '法定代表人姓名' or x == '法定代表人' or x == '法定代表人(负责人、董事长、首席代表)变更' else x)

        return df_output
    
        # 数据回写入dws数据库表内
    def dataAppend(self, df, table_name, host, database, user, password):
        # 连接数据库
        password = parse.quote_plus(password)
        SQLALCHEMY_DATABASE_URI = 'postgresql://' + user + ':'  + password + '@' + host + ':8000' + '/' + database
        engine = create_engine(SQLALCHEMY_DATABASE_URI, client_encoding='utf8')
        # 写入数据库
        df.to_sql(schema='ods', con=engine, name = table_name, if_exists='append', index=False)
    
    
    # 启动函数
    def apply(self):
        ###获取配置文件里的敏感信息，进行解密
        host = self.AES_de(self.host, self.key, self.iv)
        database = self.AES_de(self.database, self.key, self.iv)
        user = self.AES_de(self.user, self.key, self.iv)
        password = self.AES_de(self.password, self.key, self.iv)
        ### 返回客户列表
        customer_list = self.dwsConnect(host, database, user, password)
        ### 调取客户基本信息
        df_basic_info = self.customerBased(customer_list)
        ### 调取客户风险
        df_risk = self.dateBased(customer_list)
        ### 数据回写
        self.dataAppend(df_basic_info, 'tianyancha_basic_info', host, database, user, password)
        self.dataAppend(df_risk, 'tianyancha_risk', host, database, user, password)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("/home/qc/iiot_config/conf.ini")
    dws = Tianyancha(config)
    dws.apply()
    print('Success')