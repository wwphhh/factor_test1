import pandas as pd
import datetime
from sklearn import linear_model,preprocessing
import numpy as np
import itertools
from talib.abstract import *
import talib
import matplotlib.pyplot as plt


class factor_construct(object):
    def __init__(self,filename="2021-06-18上证50自己清单.xls"):
        # 设定清单文件
        self.primary_load = "raw_data/"
        df_ = self.load_code_list(filename=filename)
        index_list = df_.index.tolist()
        self.index_list = ["SH." + i + ".L2" for i in index_list]
        # 获取所有原始数据
        self.load_all_data()
        pass

    def load_code_list(self, filename):
        df_all_data = pd.read_excel(filename, dtype="str")
        df_ = df_all_data[["证券代码", "证券数量"]].set_index("证券代码")
        df_ = df_.astype("float")
        # df_short = df_short.drop(df_short[df_short.证券数量==0].index).sort_index()
        return df_

    def load_all_data(self):
        df_all_data = pd.DataFrame()
        for code in self.index_list:
            df_temp = pd.read_csv(self.primary_load + code + ".csv")
            df_temp["time"] = pd.to_datetime(df_temp["time"])
            df_temp = df_temp.set_index("time")
            df_all_data = df_all_data.append(df_temp)
        self.df_all_data = df_all_data
        print("all data has been loaded!")
        pass

    def get_primary_load(self):
        return self.primary_load

    def set_primary_load(self,primary_load):
        self.primary_load=primary_load

    def set_index_list(self,index_list):
        self.index_list = index_list

    # 活跃度因子计算，对传入的MAtch价格序列做回归处理
    def activity_factor(self,temp_list):
        x_lable = np.array([i+1 for i in range(len(temp_list))])
        y_lable = temp_list
        # 使用最大最小值归一化方法进行归一化
        min_max_scaler = preprocessing.MinMaxScaler()
        # x_lable = min_max_scaler.fit_transform(x_lable.reshape(-1,1))
        x_lable = x_lable.reshape(-1,1)
        y_lable = min_max_scaler.fit_transform(y_lable.reshape(-1,1))
        model = linear_model.LinearRegression()
        model.fit(x_lable,y_lable)
        # 计算线性回归残差
        y_predict = model.predict(X=x_lable)
        err_list = (y_lable-y_predict).reshape(-1)
        df_err = pd.DataFrame(err_list)
        # 将线性回归残差作为活跃度因子
        factor_value = np.std(df_err.diff().dropna(how="any").values)
        return factor_value
        pass

    # 成交量因子计算，输入一段时间内的成交量数据和因子的时间区间
    def volume_factor(self,templist,scale_value):
        if len(templist)<scale_value+1:
            return -1
        # 获取当前时刻的成交量均值和上一时刻的成交量均值
        mean_1=np.mean(np.array(templist[:-2]))
        mean_2 = np.mean(np.array(templist[1:-1]))
        # 获取当前时刻的成交量乖离率和上一时刻的成交量乖离率
        Deviation_1 = abs(templist[-2]-mean_1)/mean_1
        Deviation_2 = abs(templist[-1] - mean_2) / mean_2
        # 将成交量乖离率的变化率作为当前时刻的因子值
        factor_value = (Deviation_2-Deviation_1)*1.0/Deviation_1
        return factor_value

    # 每日因子值与因变量值计算
    def daily_factor(self,df_group):
        # 最小成交量滑动窗口大小
        volume_forward_low=10
        # 成交量因子不同的滑动窗口大小
        volume_forward_high=[20,30,60,120]
        # 最小活跃度因子滑动窗口大小
        activity_forward_low=20
        # 活跃度因子不同的滑动窗口大小
        activity_forward_high = [20,30,40,60]
        future_activity_10m = []
        future_activity_15m = []
        future_activity_20m = []
        future_activity_30m = []
        present_volume_factor_10m = []
        present_volume_factor_15m = []
        present_volume_factor_30m = []
        present_volume_factor_1h = []
        for i in range(len(df_group)):
            # 未来活跃度计算
            if i+activity_forward_low<=len(df_group):
                if i+activity_forward_high[0]<=len(df_group):
                    factor_value = self.activity_factor(df_group[i:i+activity_forward_high[0]]["Match"].values)
                    future_activity_10m.append(factor_value)
                else:
                    factor_value = self.activity_factor(df_group[i:]["Match"].values)
                    future_activity_10m.append(factor_value)

                if i+activity_forward_high[1]<=len(df_group):
                    factor_value = self.activity_factor(df_group[i:i+activity_forward_high[1]]["Match"].values)
                    future_activity_15m.append(factor_value)
                else:
                    factor_value = self.activity_factor(df_group[i:]["Match"].values)
                    future_activity_15m.append(factor_value)

                if i+activity_forward_high[2]<=len(df_group):
                    factor_value = self.activity_factor(df_group[i:i+activity_forward_high[2]]["Match"].values)
                    future_activity_20m.append(factor_value)
                else:
                    factor_value = self.activity_factor(df_group[i:]["Match"].values)
                    future_activity_20m.append(factor_value)

                if i+activity_forward_high[3]<=len(df_group):
                    factor_value = self.activity_factor(df_group[i:i+activity_forward_high[3]]["Match"].values)
                    future_activity_30m.append(factor_value)
                else:
                    factor_value = self.activity_factor(df_group[i:]["Match"].values)
                    future_activity_30m.append(factor_value)
                pass
            else:
                future_activity_10m.append(future_activity_10m[-1])
                future_activity_15m.append(future_activity_15m[-1])
                future_activity_20m.append(future_activity_20m[-1])
                future_activity_30m.append(future_activity_30m[-1])
                pass

            # 成交量乖离因子计算
            if i >= volume_forward_low:
                volume_list = pd.DataFrame(df_group["Volume"].values.tolist()).diff().values
                volume_list[0]=df_group["Volume"].values.tolist()[0]
                volume_list = volume_list.reshape(-1)
                if i<=volume_forward_high[0]:
                    volume_factor_value = self.volume_factor(volume_list[:i+1],i-1)
                else:
                    volume_factor_value = self.volume_factor(volume_list[i-volume_forward_high[0]:i + 1],volume_forward_high[0])
                present_volume_factor_10m.append(volume_factor_value)

                if i<=volume_forward_high[1]:
                    volume_factor_value = self.volume_factor(volume_list[:i+1],i-1)
                else:
                    volume_factor_value = self.volume_factor(volume_list[i-volume_forward_high[1]:i + 1],volume_forward_high[1])
                present_volume_factor_15m.append(volume_factor_value)

                if i<=volume_forward_high[2]:
                    volume_factor_value = self.volume_factor(volume_list[:i+1],i-1)
                else:
                    volume_factor_value = self.volume_factor(volume_list[i-volume_forward_high[2]:i + 1],volume_forward_high[2])
                present_volume_factor_30m.append(volume_factor_value)

                if i<=volume_forward_high[3]:
                    volume_factor_value = self.volume_factor(volume_list[:i+1],i-1)
                else:
                    volume_factor_value = self.volume_factor(volume_list[i-volume_forward_high[3]:i + 1],volume_forward_high[3])
                present_volume_factor_1h.append(volume_factor_value)

            else:
                present_volume_factor_10m.append(np.NAN)
                present_volume_factor_15m.append(np.NAN)
                present_volume_factor_30m.append(np.NAN)
                present_volume_factor_1h.append(np.NAN)
                pass

        df_group["future_activity_10m"]=future_activity_10m
        df_group["future_activity_15m"] = future_activity_15m
        df_group["future_activity_20m"] = future_activity_20m
        df_group["future_activity_30m"] = future_activity_30m

        df_group["volume_factor_10m"]=present_volume_factor_10m
        df_group["volume_factor_15m"] = present_volume_factor_15m
        df_group["volume_factor_30m"] = present_volume_factor_30m
        df_group["volume_factor_1h"] = present_volume_factor_1h

        return df_group

    def get_all_factor(self):
        def get_all_daily_factor(df_group):
            print(list(set(df_group["StkCode"].values))[0])
            res = df_group.groupby(pd.DatetimeIndex(df_group.index).normalize(), as_index=False, group_keys=False).apply(
                self.daily_factor)
            print(len(res))
            print(list(set(df_group["StkCode"].values))[0])
            return res

        def filter_daily_data(df_group):

            code = list(set(df_group["StkCode"].values))[0]
            print(code)

            # 去除停牌数据
            df_group = df_group[~(df_group["Match"] == 0)]

            # 获取原始数据
            df_group = df_group.reset_index()
            end_time = datetime.datetime.strptime("7:00:00", "%H:%M:%S").time()
            df_temp = df_group[df_group.time.dt.time == end_time]
            df_temp.set_index("time",inplace=True)

            # 因子计算准备相关变量
            DIF,DEA,MACD = talib.MACD(df_temp["Match"]/10000,fastperiod=12, slowperiod=26, signalperiod=9)
            up,mid,low = talib.BBANDS(df_temp["Match"]/10000,timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
            volume_SMA10 = talib.SMA(df_temp["Volume"],timeperiod=10)
            df_temp["DIF"]=DIF
            df_temp["DEA"] = DEA
            df_temp["MACD"] = MACD
            df_temp["diff_DIF"] = df_temp["DIF"].diff()
            df_temp["diff_DEA"] = df_temp["DEA"].diff()
            df_temp["diff_MACD"] = df_temp["MACD"].diff()
            df_temp["up"]=up
            df_temp["mid"] = mid
            df_temp["low"] = low
            df_temp["volume_SMA10"]=volume_SMA10

            # 收益率计算相关变量
            buy_time = datetime.datetime.strptime("6:30:00", "%H:%M:%S").time()
            df_buy_price = df_group[df_group.time.dt.time == buy_time]
            df_buy_price.set_index("time", inplace=True)
            df_temp["buy_price"] = df_buy_price["Match"].values / 10000
            df_temp["future_sell_price"] = df_temp["Open"].shift(periods=-1)
            df_temp["future_sell_price"] = df_temp["future_sell_price"]/10000

            # 数据过滤 去除NAN
            # df_temp = df_temp.dropna()


            # 成交量乖离率因子
            df_temp["volume_factor"]=(df_temp["Volume"]-df_temp["volume_SMA10"])*1.0/df_temp["volume_SMA10"]
            temp = df_temp["Match"]/df_temp["PreClose"]-1
            temp = temp/temp.__abs__()
            temp=temp.fillna(1)
            df_temp["volume_factor"] = df_temp["volume_factor"]*temp
            df_temp["volume_factor"] = df_temp["volume_factor"].shift()


            #布林线因子
            df_temp["boll_factor"] = df_temp["Match"]/df_temp["mid"]-1
            temp = df_temp["boll_factor"].shift()
            df_temp["boll_factor"] = df_temp["boll_factor"].diff()
            df_temp["boll_factor"]=df_temp["boll_factor"]/temp
            df_temp["boll_factor"] = df_temp["boll_factor"].shift()

            # MACD因子
            def MACD_Fctor(diff_DIF, diff_DEA, MACD):
                if diff_DIF > 0 and diff_DEA > 0 and MACD > 0:
                    signal = 1
                else:
                    signal = -1
                value = abs(MACD)
                return signal*value

            df_temp["MACD_Fctor"] = df_temp.apply(
                lambda row: MACD_Fctor(row["diff_DIF"], row["diff_DEA"], row["MACD"]), axis=1
            )
            df_temp["MACD_Fctor"] = df_temp["MACD_Fctor"].shift()

            # 数据过滤 去除NAN
            df_temp = df_temp.dropna()

            # 收益率计算
            df_temp["furure_return_rate"] = df_temp["future_sell_price"]/df_temp["buy_price"]-1

            print(len(df_temp))
            return df_temp
            pass

        # res = self.df_all_data.groupby(self.df_all_data["StkCode"], as_index=False, group_keys=False).apply(get_all_daily_factor)
        res = self.df_all_data.groupby(self.df_all_data["StkCode"], as_index=False, group_keys=False).apply(
            filter_daily_data)
        print(res)
        res.to_csv("all_data_with_factor.csv")


if __name__=="__main__":
    factor_ob = factor_construct()
    factor_ob.get_all_factor()


