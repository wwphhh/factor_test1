#coding = utf8
import pandas as pd
import datetime
import copy
import numpy as np
from sklearn import linear_model,preprocessing
import statsmodels.api as sm
from functools import *
import itertools
class analysis(object):
    def __init__(
            self,
            primary_load="raw_data/",
            primary_load_rankic_bycode="rank_ic_bycode/",
            factor_list=['volume_factor_10m', 'volume_factor_15m', 'volume_factor_30m', 'volume_factor_1h'],
            dependent_var=['future_activity_10m', 'future_activity_15m', 'future_activity_20m', 'future_activity_30m'],
            commission_rate = 2.5/10000,
            filename = "all_data_with_factor.csv",
            if_T0 = True
                 ):
        """
        :param primary_load: 原始数据存储主路径
        :param primary_load_rankic_bycode: IC值结果存储路径
        :param factor_list: 因子名称
        :param dependent_var: 应变量名称
        """
        self.primary_load = primary_load
        self.primary_load_rankic_bycode = primary_load_rankic_bycode
        self.factor_list = factor_list
        self.dependent_var = dependent_var
        self.set_index_list()
        self.load_all_data(filename=filename,if_T0=if_T0)
        self.commission_rate = commission_rate
        pass

    def get_commission_value(self,cost):
        commission = cost*self.commission_rate
        if commission<=5:
            return 5
        else:
            return commission

    # 获取全部数据并过滤停牌数据和尾盘集合竞价时间的数据
    def load_all_data_old(self):
        df_all_data = pd.DataFrame()
        for code in self.index_list:
            df_temp = pd.read_csv(self.primary_load + code + ".csv")
            df_temp = df_temp[~(df_temp["Match"] == 0)]
            df_temp = df_temp.dropna()
            df_temp["time"] = pd.to_datetime(df_temp["time"])
            end_time = datetime.datetime.strptime("6:57:00", "%H:%M:%S").time()
            df_temp = df_temp[(df_temp.time.dt.time <= end_time)]
            df_temp = df_temp.set_index("time")
            df_all_data = df_all_data.append(df_temp)
        self.df_all_data = df_all_data
        # print(self.df_all_data)
        return self.df_all_data

    def load_all_data(self,filename = "all_data_with_factor.csv",if_T0 = True):
        df_all_data = pd.read_csv(filename)
        df_all_data = df_all_data[~(df_all_data["Match"] == 0)]
        df_all_data = df_all_data.dropna()
        df_all_data["time"] = pd.to_datetime(df_all_data["time"])
        if if_T0:
            end_time = datetime.datetime.strptime("6:57:00", "%H:%M:%S").time()
            df_all_data = df_all_data[(df_all_data.time.dt.time <= end_time)]
        df_all_data = df_all_data.set_index("time")
        self.df_all_data = df_all_data
        # print(self.df_all_data)
        return self.df_all_data
        pass

    # 人工设定因子列名
    def set_factor_list(self,factor_list):
        if isinstance(factor_list,list):
            self.factor_list = factor_list
        elif isinstance(factor_list,str):
            self.factor_list = factor_list.split(",")
        else:
            return -1

    # 获取因子列名
    def get_factor_list(self):
        return self.factor_list

    # 人工设定因变量列名
    def set_dependent_var(self, dependent_var):
        if isinstance(dependent_var, list):
            self.dependent_var = dependent_var
        elif isinstance(dependent_var, str):
            self.dependent_var = dependent_var.split(",")
        else:
            return -1

    # 获取因变量列名
    def get_dependent_var(self):
        return self.dependent_var

    #设定清单文件
    def set_index_list(self,filename="2021-06-18上证50自己清单.xls"):
        df_all_data = pd.read_excel(filename, dtype="str")
        df_index = df_all_data[["证券代码", "证券数量"]].set_index("证券代码")
        df_index = df_index.astype("float")
        index_list = df_index.index.tolist()
        self.index_list = ["SH." + i + ".L2" for i in index_list]

    # 中位数去极值法
    def MAD_process(self, df_series):
        medium_raw = df_series.quantile(0.5)
        new_medium = ((df_series - medium_raw).abs()).quantile(0.5)
        up = ((df_series - medium_raw).abs()).quantile(0.95)
        down = ((df_series - medium_raw).abs()).quantile(0.05)
        n = max(up / new_medium, down / new_medium)
        max_range = medium_raw + n * new_medium
        min_range = medium_raw - n * new_medium
        return np.clip(df_series, min_range, max_range)

    # 计算个股因子IC， 对个股计算每日IC值
    def get_rank_ic_bycode(self):
        # 根据相关性举证找出每日的因子与因变量之间的IC值
        def calculate_daily_rankic(df_group):
            res_df = df_group[self.factor_list + self.dependent_var].corr()
            res_df = res_df[-len(self.dependent_var):][self.factor_list]
            column_list = [i[0]+"_to_"+i[1] for i in itertools.product(self.factor_list,self.dependent_var)]
            value_list = res_df.T.values.reshape(1,-1)
            res = pd.DataFrame(value_list,columns=column_list)
            return res

        def calculate_code_rankic(df_group):
            # res_df = df_group.groupby(pd.DatetimeIndex(df_group.index).normalize()).apply(calculate_daily_rankic)
            # res_df.index = [i[0] for i in res_df.index]
            # res_df.index.name = "time"
            code = list(set(df_group["StkCode"].values))[0]

            res_df = calculate_daily_rankic(df_group)
            res_df.to_csv(self.primary_load_rankic_bycode + code + ".csv", index=True)
            print(code)
            return res_df
            pass
        res_ = self.df_all_data.groupby(self.df_all_data["StkCode"]).apply(calculate_code_rankic)
        return res_

    #时间截面因子IC 对每一个时刻计算所有股票的IC值
    def get_rankic_bytime(self):
        # 根据相关性举证找出每日的因子与因变量之间的IC值
        def calculate_timely_rankic(df_group):
            res_df = df_group[self.factor_list + self.dependent_var].corr()
            res_df = res_df[-len(self.dependent_var):][self.factor_list]
            column_list = [i[0] + "_to_" + i[1] for i in itertools.product(self.factor_list, self.dependent_var)]
            value_list = res_df.T.values.reshape(1, -1)
            res = pd.DataFrame(value_list, columns=column_list)
            return res

        # 按时间分组并计算截面IC值
        res_df = self.df_all_data.groupby(self.df_all_data.index).apply(calculate_timely_rankic)
        res_df.index = [i[0] for i in res_df.index]
        res_df.index.name = "time"
        res_df.to_csv("rank_ic_timely.csv")
        print(res_df)
        return res_df

    # 对结果的IC矩阵进行分析
    def analysis_rankic_res(self,df_ic_values,to_file=False,**kwargs):
        index_list = df_ic_values.columns
        analysis_res_dic = {
            "IC值大于0比例": [],
            "相邻IC同正比例": [],
            "IC均值": [],
            "IC标准差": [],
            "IC_IR": [],
        }
        def calcul(df_):
            analysis_res_dic["IC值大于0比例"].append(len([i for i in df_.values if i >0])*1.0/len(df_))
            analysis_res_dic["相邻IC同正比例"].append(len([i for i in range(len(df_)-1) if df_.values[i]>0 and df_[i+1]>0])*1.0/(len(df_)-1))
            analysis_res_dic["IC均值"].append(df_.values.mean())
            analysis_res_dic["IC标准差"].append(df_.std())
            analysis_res_dic["IC_IR"].append(df_.values.mean()/df_.std())
        df_ic_values.apply(calcul)
        res_df = pd.DataFrame(analysis_res_dic,index=index_list)
        if to_file:
            if "filename" not in kwargs:
                exit(-1)
                return -1
            code = df_ic_values.index[0][0]
            res_df.to_csv(kwargs["filename"]+code+".csv",encoding="GBK")
            print(code+" ic analysis done!")
        return res_df
        pass

    # 对股票截面的IC序列分析
    def analysis_rankic_bycode(self,fileload="analysis_rank_ic_bycode/"):
        res_ = self.get_rank_ic_bycode()
        res_.groupby(level=0).apply(self.analysis_rankic_res,to_file=True,filename = fileload)

    # 对时间截面上的IC序列进行分析
    def analysis_rankic_bytime(self,filename = "analysis_rankic_bytime.csv"):
        res_ = self.get_rankic_bytime()
        analysis_res_df = self.analysis_rankic_res(res_)
        analysis_res_df.to_csv(filename,encoding="GBK")
        pass

    # 单周期因子收益率分析
    def factor_return_analysis_single(self,future_rank = 4, return_factor = False,to_file = True,**kwargs):
        if "method" in kwargs:
            method = kwargs["method"]
        else:
            method = "linear"
        # 计算未来收益
        def calculate_return_factor(df_group):
            res_list = []
            match_list = df_group['Match'].values
            for i in range(len(match_list)):
                if i+future_rank>=len(match_list):
                    res_list.append(0)
                else:
                    # temp_res = match_list[i+future_rank]*1.0/match_list[i]-1
                    temp_res = ((match_list[i+future_rank]-match_list[i])*100.0-self.get_commission_value((match_list[i+future_rank]+match_list[i])*100))/(match_list[i]*100+self.get_commission_value(match_list[i]*100))
                    res_list.append(temp_res)
            df_group["future_return_rate_"+str(future_rank)]=res_list
            return df_group
            pass

        # 计算因子与未来收益的相关性
        def calculate_timely_factor_return_corr(df_group):
            df_corr = df_group[self.factor_list+["future_return_rate_"+str(future_rank)]].corr()
            column_list = [i+"_to_future_return_rate_"+str(future_rank) for i in self.factor_list]
            res_df = pd.DataFrame(df_corr[-1:][self.factor_list])
            res_df.columns=column_list
            return res_df
            pass

        # 线性回归计算因子收益率
        def calculate_timely_factor_return(df_group):
            res_dic= {}
            for factor_name in self.factor_list:
                x_lable = self.MAD_process(df_group[factor_name])
                min_max_scaler = preprocessing.MinMaxScaler()
                x_lable = min_max_scaler.fit_transform(x_lable.values.reshape(-1, 1))
                y_lable = df_group["future_return_rate_"+str(future_rank)].values.reshape(-1,1)
                model = linear_model.LinearRegression()
                model.fit(x_lable, y_lable)
                factor_return_value = model.coef_.item()
                key_str = factor_name+"_to_future_return_rate_"+str(future_rank)
                if key_str not in res_dic:
                    res_dic[key_str]=[]
                res_dic[key_str].append(factor_return_value)
            factor_return_value_df = pd.DataFrame(res_dic)
            return factor_return_value_df
            pass

        # # 因变量列名
        # if return_factor:
        #     if len(kwargs)==0:
        #         return -1
        #     else:
        #         pass
        # 将数据按照股票代码分组，并计算每个时刻下的未来收益率
        def calculate_return_factor_bycode(df_group):
            res_ = df_group.groupby(pd.DatetimeIndex(df_group.index).normalize()).apply(calculate_return_factor)
            res_ = res_[~(res_["future_return_rate_" + str(future_rank)] == 0)]
            return res_
            pass
        df_all_data = self.df_all_data.groupby(self.df_all_data["StkCode"],as_index=False,group_keys=False).apply(calculate_return_factor_bycode)

        if method=="corr":
            res_df = df_all_data.groupby(df_all_data.index).apply(calculate_timely_factor_return_corr)
        elif method=="linear":
            res_df = df_all_data.groupby(df_all_data.index).apply(calculate_timely_factor_return)
        else:
            return -1
        res_df.index = [i[0] for i in res_df.index]
        res_df.index.name = "time"
        if to_file:
            res_df.to_csv("factor_return_rate_timely_"+str(future_rank)+"_"+method+".csv")
        print(res_df)
        return res_df
        pass

    # 多周期因子收益率检验
    def factor_return_analysis_multi(self,future_rank_list=[4,10,20,30,60],**kwargs):

        if "method" in kwargs:
            method = kwargs["method"]
        else:
            method = "linear"

        df_all_data = []
        for future_rank in future_rank_list:
            print(future_rank)
            df_temp = self.factor_return_analysis_single(future_rank=future_rank,to_file=False)
            df_all_data.append(df_temp)
        df_all_data = pd.concat(df_all_data,join="inner",axis=1)
        # df_temp.append()
        print(df_all_data)
        df_all_data.to_csv("factor_return_rate_timely_"+method+".csv")

    # 已有收益率下的因子收益率检验
    def factor_return_analysis_stable(self,return_columns,method = "linear"):
        if isinstance(return_columns, list):
            return_columns = return_columns
        elif isinstance(return_columns, str):
            return_columns = return_columns.split(",")
        else:
            return -1

        self.set_dependent_var(return_columns)

        # 因子与收益率的IC相关性检验
        def calculate_timely_factor_return_corr(df_group):
            df_corr = df_group[self.factor_list+self.dependent_var].corr()
            column_list = [i[0] + "_to_" + i[1] for i in itertools.product(self.factor_list, self.dependent_var)]
            res_df = pd.DataFrame(df_corr[-len(self.dependent_var):][self.factor_list])
            value_list = res_df.T.values.reshape(1, -1)
            res_df = pd.DataFrame(value_list, columns=column_list)
            return res_df
            pass

            # 线性回归计算因子收益率
        def calculate_timely_factor_return_linesr(df_group):
            res_dic = {}
            for factor_name,return_columns_name in itertools.product(self.factor_list, self.dependent_var):
                x_lable = self.MAD_process(df_group[factor_name])
                min_max_scaler = preprocessing.MinMaxScaler()
                x_lable = min_max_scaler.fit_transform(x_lable.values.reshape(-1, 1))
                y_lable = df_group[return_columns_name].values.reshape(-1, 1)
                model = sm.OLS(y_lable,x_lable)
                result = model.fit()

                # 计算并存储因子收益率
                factor_return_value = result.params.item()
                key_str = factor_name + "_to_" + return_columns_name+"_factor_return_rate"
                if key_str not in res_dic:
                    res_dic[key_str] = []
                res_dic[key_str].append(factor_return_value)

                # 计算并存储回归检验量p值
                p_value = result.pvalues.item()
                key_str = factor_name + "_to_" + return_columns_name + "_p_value"
                if key_str not in res_dic:
                    res_dic[key_str] = []
                res_dic[key_str].append(p_value)

                # 计算并存储回归检验量t值
                t_value = result.tvalues.item()
                key_str = factor_name + "_to_" + return_columns_name + "_t_value"
                if key_str not in res_dic:
                    res_dic[key_str] = []
                res_dic[key_str].append(t_value)

                # 计算并存储回归检验量R方
                r_square = result.rsquared.item()
                key_str = factor_name + "_to_" + return_columns_name + "_r_square"
                if key_str not in res_dic:
                    res_dic[key_str] = []
                res_dic[key_str].append(r_square)

            factor_return_res_df = pd.DataFrame(res_dic)
            return factor_return_res_df
            pass

        if method=="corr":
            # 因子与收益率相关性分析
            res_df = self.df_all_data.groupby(self.df_all_data.index).apply(calculate_timely_factor_return_corr)
            res_df.index = [i[0] for i in res_df.index]
            res_df.index.name = "time"
            res_df.to_csv("factor_return_rate_timely_IC.csv")
            print(res_df)
        elif method=="linear":
            res_df = self.df_all_data.groupby(self.df_all_data.index).apply(calculate_timely_factor_return_linesr)
            res_df.index = [i[0] for i in res_df.index]
            res_df.index.name = "time"
            res_df.to_csv("factor_return_rate_timely_linear.csv")
            print(res_df)
            pass
        pass

    # 因子收益率检验
    def factor_return_analysis(self,with_return=False,**kwargs):
        if with_return:
            if "return_columns" in kwargs:
                return_columns = kwargs["return_columns"]
                if "method" in kwargs:
                    method = kwargs["method"]
                    self.factor_return_analysis_stable(return_columns,method=method)
                else:
                    self.factor_return_analysis_stable(return_columns)
            else:
                return -1
        else:
            if "method" in kwargs:
                method = kwargs["method"]
                self.factor_return_analysis_multi(method=method)
            else:
                self.factor_return_analysis_multi()
        pass

    # 分层检验
    def monotonicity_alalysis(self, level_num = 5,if_T0=True):
        num_of_code = len(self.index_list)
        len_of_group = int(num_of_code/level_num)
        # 即时收益
        def calculate_timely_return(df_group):
            # 计算收益时不包括手续费
            pre_match = df_group["Match"].shift()
            df_group["return"] = df_group["Match"] / pre_match - 1

            # 计算收益时包括手续费
            # match_list = df_group["Match"].values.tolist()
            # return_list = [
            #     ((match_list[i] - match_list[i-1]) * 100.0 - self.get_commission_value((match_list[i]+match_list[i-1]) * 100)) /
            #     (match_list[i-1] * 100 + self.get_commission_value(match_list[i-1] * 100))
            #                for i in range(1,len(match_list))
            #                ]
            # return_list.insert(0,np.NAN)
            # df_group["return"]=return_list
            return df_group
            pass
        # 因子值排序
        def factor_value_rank(df_group):
            for factor in self.factor_list:
                df_group = df_group.sort_values(axis=0,by=factor)
                df_group[factor+"_rank"]=range(1,len(df_group)+1)
            return df_group
            pass
        # 数值对齐 获取上一时刻的因子排名
        def data_align(df_group):
            def daily_data(df_daily):
                for factor in self.factor_list:
                    df_daily[factor+"_rank_pre"]=df_daily[factor+"_rank"].shift()
                df_daily = df_daily.dropna()
                return df_daily
            if if_T0:
                res_df = df_group.groupby(pd.DatetimeIndex(df_group.index).normalize(),as_index=False,group_keys=False).apply(daily_data)
            else:
                res_df = daily_data(df_group)
            return res_df

        # 分组收益
        def calculate_portfolio_return(df_group):
            res_dic = {}
            for factor in self.factor_list:
                for i in range(level_num):
                    res_dic[factor + "_group_" + str(i + 1)] = []
            for factor in self.factor_list:
                df_temp = df_group.sort_values(axis=0,by=factor+"_rank_pre")
                return_list = df_temp["return"].values
                for i in range(level_num):
                    if i==level_num-1:
                        res_dic[factor + "_group_" + str(i + 1)].append(
                            np.mean(return_list[len_of_group * i:]))
                    else:
                        res_dic[factor + "_group_"+str(i+1)].append(np.mean(return_list[len_of_group*i:len_of_group*(i+1)]))
            df_return = pd.DataFrame(res_dic)
            return df_return

        # 获取单个股票每日数据并计算每一个时间戳（日内或日间）的收益率
        def get_return_bycode(df_group):
            if if_T0:
                # 每一时刻下相对于上一时刻收益率计算
                res_df = df_group.groupby(pd.DatetimeIndex(df_group.index).normalize(), as_index=False,
                                        group_keys=False).apply(calculate_timely_return)
            else:
                # 计算每日收益率
                res_df = calculate_timely_return(df_group)
            return res_df
            pass

        # 获取所有股票每日数据并计算因子排名和每一时刻的收益率
        df_all_data = self.df_all_data.groupby(self.df_all_data["StkCode"],as_index=False,group_keys=False).apply(get_return_bycode)

        # 计算因子排名
        res_df = df_all_data.groupby(pd.DatetimeIndex(df_all_data.index),as_index=False,group_keys=False).apply(factor_value_rank)

        # 数据对齐，确保每一时间戳可获取上一时间戳的因子排名
        res_df = res_df.groupby(res_df["StkCode"],as_index=False,group_keys=False).apply(data_align)

        # 计算每一个时间戳下的分组收益
        res_df = res_df.groupby(pd.DatetimeIndex(res_df.index)).apply(calculate_portfolio_return)

        res_df.index = [i[0] for i in res_df.index]
        res_df.index.name = "time"
        # print(res_df)
        # exit()
        # res_df.to_csv("分层.csv")
        # 计算分层净值
        for column in res_df.columns:
            res_df[column]=res_df[column]+1
        res_list = []
        for i in range(len(res_df),0,-1):
            np_temp = res_df.iloc[:i].values
            temp = reduce(lambda x,y:x*y,np_temp)
            res_list.insert(0,temp)
            pass
        res_df = pd.DataFrame(index=res_df.index,columns=res_df.columns,data=np.array(res_list))
        print(res_df)
        res_df.to_csv("分层净值.csv")
        pass


if __name__=="__main__":
    filename = "all_data_with_factor.csv"
    analysis_test = analysis(filename=filename)
    analysis_test.load_all_data(filename=filename,if_T0=False)
    # analysis_test.load_all_data(filename="all_data_with_factor_2021.7.7.csv")

    factor_list = ["volume_factor","boll_factor","MACD_Fctor"]
    return_columns = ["furure_return_rate"]
    analysis_test.set_factor_list(factor_list=factor_list)
    analysis_test.set_dependent_var(return_columns)
    # 因子个股截面上的IC结果分析
    # analysis_test.analysis_rankic_bycode()
    # exit()
    # 因子时间截面上的IC结果分析
    # analysis_test.analysis_rankic_bytime()
    # exit()
    # analysis_test.factor_return_analysis()
    # analysis_test.factor_return_analysis(with_return=True,return_columns="furure_return_rate",method="linear")
    # exit()
    # analysis_test.factor_return_analysis_multi()
    analysis_test.monotonicity_alalysis(if_T0=False)
    # analysis_test.monotonicity_alalysis()