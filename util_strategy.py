import numpy as np
import pandas as pd
import random
from random import randint
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

def get_median_regression(df,y,xlst):
    yxmode = y+' ~ '+xlst[0]
    for i in range(1,len(xlst)):
        yxmode = yxmode + ' + ' + xlst[i]
    model = smf.quantreg(yxmode, df)
    result = model.fit(q=0.5, weights=np.exp(df['y'].values)-1.0)
    return result

def get_params_GoldChange():
    X = {}
    Y = {}
    Z = {}
    columns = ['y']
    columns.append('est')
    # change range, and column counts
    for j in range(-2,3):
        X[j]=[]
        Y[j]=[]
        Z[j]=pd.DataFrame(columns=columns)
    
    
    for i in range(100):
        game_value = get_game_value()
        last_yield = -1
        #
        sum_log_yield = 5.0*np.log(101.0)
        for j in range(150):
            picked, mining_time_first, mining_time = sample_picked_once_with_time(game_value)
            if j==0:
                change_of_gold = -1
                change_of_diamond = -1
                fifty_yields = []
                for k in range(50):
                    fifty_yields.append(sample_yield_once(game_value))
                last_lowest = 1e9
                last_highest = -1e9
                exact_diamond = -1
            else:
                fifty_yields = []
                change_of_gold = picked[2] - last_gold
                if change_of_gold<-3:
                    change_of_gold = -2
                elif change_of_gold<=-1:
                    change_of_gold = -1
                elif change_of_gold<=0:
                    change_of_gold = 0
                elif change_of_gold<=3:
                    change_of_gold = 1
                else:
                    change_of_gold = 2
                change_of_diamond = picked[4] - last_diamond
                if change_of_diamond<-1:
                    change_of_diamond = -1
                elif change_of_diamond <=1:
                    change_of_diamond = 0
                else:
                    change_of_diamond = 1
                if picked[4] == last_diamond:
                    exact_diamond = last_diamond
                else:
                    exact_diamond = -1
            mining_time_bak = mining_time
            if mining_time<=400:
                mining_time=-1
            this_lowest = 12345678
            this_highest = -12345678
            this_yield = get_yield_by_picked(picked)
            #
            # change key words
            if j>0:
                #X[change_of_gold].append(sum_log_yield/(j+5))
                #Y[change_of_gold].append(this_yield)
                Z[change_of_gold].loc[len(Z[change_of_gold])] = [np.log(this_yield+1.0), sum_log_yield/(j+5)]
            #
            last_gold = picked[2]
            last_diamond = picked[4]
            last_lowest = this_lowest
            last_highest = this_highest
            last_yield = this_yield
            sum_log_yield += np.log( this_yield+1.0 )
    
    
    a1 = []
    a2 = []
    b = []
    # change range / num of params
    for j in range(-2,3):
        result = get_median_regression(Z[j], 'y', ['est'])
        a1.append(result.params[1])
        b.append(result.params[0])
    print('a1=',a1)
    print('b=',b)

def GoldChange(last_total_bid, last_yield, s1, s2, s3, change_of_gold, mining_time, mining_time_first, change_of_diamond,\
                                                                      fifty_yields, lowest, highest, exact_diamond):
    a1= [0.8581076022919794, 0.9627071555334888, 1.1739043709321806, 1.03999552755069, 1.120450426807287]
    b= [-0.12669073976834155, -0.27754921806799504, -0.8929212093298685, 0.21310114713370587, 0.6205073635535073]
    if last_yield == -1:
        s1=5
        s2=5.0*np.log(101.0)
        res = np.log(101.0)
    else:
        s1 = s1 + 1
        now = np.log( last_yield+1 )
        s2 = s2 + now
        x = s2/s1
        state = change_of_gold+2
        res = a1[state] * x + b[state]
    bid = np.exp(res)-1.0
    #bid = int((np.exp(s2/s1)-1.0)/7 + 0.5)
    return (bid,s1,s2,s3)

def get_params_GoldChange2():
    X = {}
    Y = {}
    Z = {}
    columns = ['y']
    columns.append('est')
    # change range, and column counts
    for j in range(-2,3):
        for k in range(-2,3):
            Z[(j,k)]=pd.DataFrame(columns=columns)
    
    
    for i in range(100):
        game_value = get_game_value()
        last_yield = -1
        #
        sum_log_yield = 5.0*np.log(101.0)
        for j in range(150):
            picked, mining_time_first, mining_time = sample_picked_once_with_time(game_value)
            if j==0:
                change_of_gold = 0
                last_change_of_gold = 0
                change_of_diamond = -1
                fifty_yields = []
                for k in range(50):
                    fifty_yields.append(sample_yield_once(game_value))
                last_lowest = 1e9
                last_highest = -1e9
                exact_diamond = -1
            else:
                fifty_yields = []
                last_change_of_gold = change_of_gold
                change_of_gold = picked[2] - last_gold
                if change_of_gold<-3:
                    change_of_gold = -2
                elif change_of_gold<=-1:
                    change_of_gold = -1
                elif change_of_gold<=0:
                    change_of_gold = 0
                elif change_of_gold<=3:
                    change_of_gold = 1
                else:
                    change_of_gold = 2
                change_of_diamond = picked[4] - last_diamond
                if change_of_diamond<-1:
                    change_of_diamond = -1
                elif change_of_diamond <=1:
                    change_of_diamond = 0
                else:
                    change_of_diamond = 1
                if picked[4] == last_diamond:
                    exact_diamond = last_diamond
                else:
                    exact_diamond = -1
            mining_time_bak = mining_time
            if mining_time<=400:
                mining_time=-1
            this_lowest = 12345678
            this_highest = -12345678
            this_yield = get_yield_by_picked(picked)
            #
            # change key words
            if j>0:
                #X[change_of_gold].append(sum_log_yield/(j+5))
                #Y[change_of_gold].append(this_yield)
                Z[(last_change_of_gold,change_of_gold)].loc[len(Z[(last_change_of_gold,change_of_gold)])] = [np.log(this_yield+1.0), sum_log_yield/(j+5)]
            #
            last_gold = picked[2]
            last_diamond = picked[4]
            last_lowest = this_lowest
            last_highest = this_highest
            last_yield = this_yield
            sum_log_yield += np.log( this_yield+1.0 )
    
    
    a1 = []
    a2 = []
    b = []
    # change range / num of params
    for j in range(-2,3):
        for k in range(-2,3):
            result = get_median_regression(Z[(j,k)], 'y', ['est'])
            a1.append(result.params[1])
            b.append(result.params[0])
    print('a1=',a1)
    print('b=',b)

def GoldChange2(last_total_bid, last_yield, s1, s2, s3, change_of_gold, mining_time, mining_time_first, change_of_diamond,\
                                                                      fifty_yields, lowest, highest, exact_diamond):
    a1= [0.39010361873851634, 0.8287051153523244, 0.8353117602785606, 0.8563855754670773, 0.9893560967213415, 0.9313793298931364, 1.032080434261762, 1.0672069069201, 1.061515604696005, 0.9671970191437969, 1.2078171058747154, 1.2669232546026572, 1.215005224496484, 1.157186352380253, 0.9231155542099077, 0.7472132390100856, 0.9582736015691576, 1.088674654041898, 0.9738366456564428, 0.9414360151104004, 0.8096458158455144, 0.9068124619276456, 1.015373116444744, 1.0759310232919101, 1.0348801984124825]
    b= [1.7257034720164222, -0.20081901877573194, 0.011131611395929771, 0.6752711008729847, 1.0138639760258155, -0.6856506867873977, -0.981756442816646, -0.7544512096732583, -0.008014879641436806, 1.1984433297850223, -1.8288871266230429, -1.851927502946489, -1.1933565999894757, -0.22238136600572964, 1.516374792184146, 0.25131535244179776, -0.2245622697725508, -0.11288242492659024, 0.783576038883392, 1.5396440348274194, 0.3363223466371341, 0.6782520931802396, 0.7028604633746882, 0.6973227973562823, 1.3319534707365117]
    if last_yield == -1:
        s1=5
        s2=5.0*np.log(101.0)
        res = np.log(101.0)
        s3=2
    else:
        s1 = s1 + 1
        now = np.log( last_yield+1 )
        s2 = s2 + now
        x = s2/s1
        now_state = change_of_gold+2
        state = s3*5+now_state
        res = a1[state] * x + b[state]
        s3=now_state
    bid = np.exp(res)-1.0
    #bid = int((np.exp(s2/s1)-1.0)/7 + 0.5)
    return (bid,s1,s2,s3)

def get_params_DiamondExact():
    X = {}
    Y = {}
    Z = {}
    columns = ['y']
    columns.append('est')
    # change range, and column counts
    for j in range(-1,13):
        X[j]=[]
        Y[j]=[]
        Z[j]=pd.DataFrame(columns=columns)
    
    
    for i in range(100):
        game_value = get_game_value()
        last_yield = -1
        #
        sum_log_yield = 5.0*np.log(101.0)
        for j in range(150):
            picked, mining_time_first, mining_time = sample_picked_once_with_time(game_value)
            if j==0:
                change_of_gold = -1
                change_of_diamond = -1
                fifty_yields = []
                for k in range(50):
                    fifty_yields.append(sample_yield_once(game_value))
                last_lowest = 1e9
                last_highest = -1e9
                exact_diamond = -1
            else:
                fifty_yields = []
                change_of_gold = picked[2] - last_gold
                if change_of_gold<-3:
                    change_of_gold = -2
                elif change_of_gold<=-1:
                    change_of_gold = -1
                elif change_of_gold<=0:
                    change_of_gold = 0
                elif change_of_gold<=3:
                    change_of_gold = 1
                else:
                    change_of_gold = 2
                change_of_diamond = picked[4] - last_diamond
                if change_of_diamond<-1:
                    change_of_diamond = -1
                elif change_of_diamond <=1:
                    change_of_diamond = 0
                else:
                    change_of_diamond = 1
                if picked[4] == last_diamond:
                    exact_diamond = last_diamond
                else:
                    exact_diamond = -1
            mining_time_bak = mining_time
            if mining_time<=400:
                mining_time=-1
            this_lowest = 12345678
            this_highest = -12345678
            this_yield = get_yield_by_picked(picked)
            #
            # change key words
            if j>0:
                #X[change_of_diamond].append(sum_log_yield/(j+5))
                #Y[change_of_diamond].append(this_yield)
                Z[exact_diamond].loc[len(Z[exact_diamond])] = [np.log(this_yield+1.0), sum_log_yield/(j+5)]
            #
            last_gold = picked[2]
            last_diamond = picked[4]
            last_lowest = this_lowest
            last_highest = this_highest
            last_yield = this_yield
            sum_log_yield += np.log( this_yield+1.0 )
    
    
    a1 = []
    a2 = []
    b = []
    # change range / num of params
    for j in range(-1,13):
        if len(Z[j])<2:
            print(j)
            continue
        result = get_median_regression(Z[j], 'y', ['est'])
        a1.append(result.params[1])
        b.append(result.params[0])
    print('a1=',a1)
    print('b=',b)

def DiamondExact(last_total_bid, last_yield, s1, s2, s3, change_of_gold, mining_time, mining_time_first, change_of_diamond,\
                                                                      fifty_yields, lowest, highest, exact_diamond):
    a1= [0.8200153344834911, 0.4962738204185827, 0.45293649271160236, 0.6837833424288545, 0.7342797606813924, 0.6519039087083869, 0.7096669517095467]
    b= [1.0341385448766547, 1.6725482927754174, 2.8383819294492962, 2.5647644146042654, 2.9601266881294337, 4.486777800066033, 4.747021983805747]
    if last_yield == -1:
        s1=5
        s2=5.0*np.log(101.0)
        res = np.log(101.0)
    else:
        s1 = s1 + 1
        now = np.log( last_yield+1 )
        s2 = s2 + now
        x = s2/s1
        # change here
        state = min(exact_diamond,5)+1
        res = a1[state] * x + b[state]
    bid = np.exp(res)-1.0
    #bid = int((np.exp(s2/s1)-1.0)/7 + 0.5)
    return (bid,s1,s2,s3)

def get_params_MiningTime():
    X = {}
    Y = {}
    Z = {}
    columns = ['y']
    columns.append('est')
    columns.append('attr')
    # change range, and column counts
    for j in range(0,2):
        X[j]=[]
        Y[j]=[]
        Z[j]=pd.DataFrame(columns=columns)
    
    
    for i in range(100):
        game_value = get_game_value()
        last_yield = -1
        #
        sum_log_yield = 5.0*np.log(101.0)
        for j in range(150):
            picked, mining_time_first, mining_time = sample_picked_once_with_time(game_value)
            if j==0:
                change_of_gold = -1
                change_of_diamond = -1
                fifty_yields = []
                for k in range(50):
                    fifty_yields.append(sample_yield_once(game_value))
                last_lowest = 1e9
                last_highest = -1e9
                exact_diamond = -1
            else:
                fifty_yields = []
                change_of_gold = picked[2] - last_gold
                if change_of_gold<-3:
                    change_of_gold = -2
                elif change_of_gold<=-1:
                    change_of_gold = -1
                elif change_of_gold<=0:
                    change_of_gold = 0
                elif change_of_gold<=3:
                    change_of_gold = 1
                else:
                    change_of_gold = 2
                change_of_diamond = picked[4] - last_diamond
                if change_of_diamond<-1:
                    change_of_diamond = -1
                elif change_of_diamond <=1:
                    change_of_diamond = 0
                else:
                    change_of_diamond = 1
                if picked[4] == last_diamond:
                    exact_diamond = last_diamond
                else:
                    exact_diamond = -1
            mining_time_bak = mining_time
            if mining_time<=400:
                mining_time=-1
            this_lowest = 12345678
            this_highest = -12345678
            this_yield = get_yield_by_picked(picked)
            #
            # change key words
            if mining_time==-1:
                #X[change_of_diamond].append(sum_log_yield/(j+5))
                #Y[change_of_diamond].append(this_yield)
                Z[0].loc[len(Z[0])] = [np.log(this_yield+1.0), sum_log_yield/(j+5), mining_time]
            else:
                Z[1].loc[len(Z[1])] = [np.log(this_yield+1.0), sum_log_yield/(j+5), mining_time]
            #
            last_gold = picked[2]
            last_diamond = picked[4]
            last_lowest = this_lowest
            last_highest = this_highest
            last_yield = this_yield
            sum_log_yield += np.log( this_yield+1.0 )
    
    
    a1 = []
    a2 = []
    b = []
    # change range / num of params
    for j in range(0,2):
        if j==0:
            result = get_median_regression(Z[j], 'y', ['est'])
        else:
            result = get_median_regression(Z[j], 'y', ['est', 'attr'])
        a1.append(result.params[1])
        if j==0:
            a2.append(0.0)
        else:
            a2.append(result.params[2])
        b.append(result.params[0])
    print('a1=',a1)
    print('a2=',a2)
    print('b=',b)

def MiningTime(last_total_bid, last_yield, s1, s2, s3, change_of_gold, mining_time, mining_time_first, change_of_diamond,\
                                                                      fifty_yields, lowest, highest, exact_diamond):
    a1= [0.6157641120555397, 0.4962013518127532]
    a2= [0.0, 0.001106837372803644]
    b= [1.6556802644058175, 3.0719291945758376]
    if last_yield == -1:
        s1=5
        s2=5.0*np.log(101.0)
        res = np.log(101.0)
    else:
        s1 = s1 + 1
        now = np.log( last_yield+1 )
        s2 = s2 + now
        x = s2/s1
        x2 = mining_time
        # change here
        if mining_time==-1:
            state=0
        else:
            state=1
        res = a1[state] * x + a2[state] * x2 + b[state]
    bid = np.exp(res)-1.0
    #bid = int((np.exp(s2/s1)-1.0)/7 + 0.5)
    return (bid,s1,s2,s3)

def get_params_MiningFirst():
    X = {}
    Y = {}
    Z = {}
    columns = ['y']
    columns.append('est')
    columns.append('attr')
    # change range, and column counts
    for j in range(0,2):
        X[j]=[]
        Y[j]=[]
        Z[j]=pd.DataFrame(columns=columns)
    
    
    for i in range(100):
        game_value = get_game_value()
        last_yield = -1
        #
        sum_log_yield = 5.0*np.log(101.0)
        for j in range(150):
            picked, mining_time_first, mining_time = sample_picked_once_with_time(game_value)
            if j==0:
                change_of_gold = -1
                change_of_diamond = -1
                fifty_yields = []
                for k in range(50):
                    fifty_yields.append(sample_yield_once(game_value))
                last_lowest = 1e9
                last_highest = -1e9
                exact_diamond = -1
            else:
                fifty_yields = []
                change_of_gold = picked[2] - last_gold
                if change_of_gold<-3:
                    change_of_gold = -2
                elif change_of_gold<=-1:
                    change_of_gold = -1
                elif change_of_gold<=0:
                    change_of_gold = 0
                elif change_of_gold<=3:
                    change_of_gold = 1
                else:
                    change_of_gold = 2
                change_of_diamond = picked[4] - last_diamond
                if change_of_diamond<-1:
                    change_of_diamond = -1
                elif change_of_diamond <=1:
                    change_of_diamond = 0
                else:
                    change_of_diamond = 1
                if picked[4] == last_diamond:
                    exact_diamond = last_diamond
                else:
                    exact_diamond = -1
            mining_time_bak = mining_time
            if mining_time<=400:
                mining_time=-1
            this_lowest = 12345678
            this_highest = -12345678
            this_yield = get_yield_by_picked(picked)
            #
            # change key words
            if mining_time_first==-1:
                #X[change_of_diamond].append(sum_log_yield/(j+5))
                #Y[change_of_diamond].append(this_yield)
                Z[0].loc[len(Z[0])] = [np.log(this_yield+1.0), sum_log_yield/(j+5), mining_time_first]
            else:
                Z[1].loc[len(Z[1])] = [np.log(this_yield+1.0), sum_log_yield/(j+5), mining_time_first]
            #
            last_gold = picked[2]
            last_diamond = picked[4]
            last_lowest = this_lowest
            last_highest = this_highest
            last_yield = this_yield
            sum_log_yield += np.log( this_yield+1.0 )
    
    
    a1 = []
    a2 = []
    b = []
    # change range / num of params
    for j in range(1,2):
        result = get_median_regression(Z[j], 'y', ['est', 'attr'])
        a1.append(result.params[1])
        a2.append(result.params[2])
        b.append(result.params[0])
    print('a1=',a1)
    print('a2=',a2)
    print('b=',b)

def MiningFirst(last_total_bid, last_yield, s1, s2, s3, change_of_gold, mining_time, mining_time_first, change_of_diamond,\
                                                                      fifty_yields, lowest, highest, exact_diamond):
    a1= [0.5102471415839742]
    a2= [0.006381517467069073]
    b= [1.5563865140617343]
    if last_yield == -1:
        s1=5
        s2=5.0*np.log(101.0)
        res = np.log(101.0)
    else:
        s1 = s1 + 1
        now = np.log( last_yield+1 )
        s2 = s2 + now
        x = s2/s1
        x2 = mining_time_first
        # change here
        state = 0
        res = a1[state] * x + a2[state] * x2 + b[state]
    bid = np.exp(res)-1.0
    #bid = int((np.exp(s2/s1)-1.0)/7 + 0.5)
    return (bid,s1,s2,s3)

def LowHigh(last_total_bid, last_yield, s1, s2, s3, change_of_gold, mining_time, mining_time_first, change_of_diamond,\
                                                                      fifty_yields, lowest, highest, exact_diamond):
    if last_yield == -1:
        s1=5
        s2=5.0*np.log(101.0)
        res = np.log(101.0)
    else:
        s1 = s1 + 1
        now = np.log( last_yield+1 )
        s2 = s2 + now
        x = s2/s1
        res = x
    bid = np.exp(res)-1.0
    if lowest!=-1 and highest==-1:
        bid = lowest
    elif lowest==-1 and highest!=-1:
        bid = highest
    #bid = int((np.exp(s2/s1)-1.0)/7 + 0.5)
    return (bid,s1,s2,s3)

def Fifty(last_total_bid, last_yield, s1, s2, s3, change_of_gold, mining_time, mining_time_first, change_of_diamond,\
                                                                      fifty_yields, lowest, highest, exact_diamond):
    if last_yield == -1:
        s1=50.0
        s2=0.0
        for i in range(50):
            s2=s2+np.log( fifty_yields[i] +1)
    else:
        s1 = s1 + 1
        now = np.log( last_yield+1 )
        s2 = s2 + now
    res = s2/s1
    bid = np.exp(res)-1.0
    #bid = int((np.exp(s2/s1)-1.0)/7 + 0.5)
    return (bid,s1,s2,s3)

def NoOccu(last_total_bid, last_yield, s1, s2, s3, change_of_gold, mining_time, mining_time_first, change_of_diamond,\
                                                                      fifty_yields, lowest, highest, exact_diamond):
    if last_yield == -1:
        s1=5
        s2=5.0*np.log(101.0)
    else:
        s1 = s1 + 1
        now = np.log( last_yield+1 )
        s2 = s2 + now
    res = s2/s1
    bid = np.exp(res)-1.0
    #bid = int((np.exp(s2/s1)-1.0)/7 + 0.5)
    return (bid,s1,s2,s3)

def generate_dpoints(num_steps, num_samples=100, lowlst = [0.0], highlst = [100.0], shiftlst = [0.0], cilst = [2.0]):
    columns = ['median']
    for j in range(1,num_steps+1):
        strave = 'logave'+str(j)
        strstd = 'logavex'+str(j)
        for low_bound in lowlst:
            for high_bound in highlst:
                for shift_value in shiftlst:
                    for ci_value in cilst:
                        now_low = round(low_bound,2)
                        now_high = round(high_bound,2)
                        now_shift = round(shift_value,2)
                        now_ci = round(ci_value,2)
                        suff = '_' + str(now_low).replace('.', 'x') + '_' + str(now_high).replace('.', 'x') + \
                                '_' + (str(now_shift).replace('.', 'x')).replace('-', 'y') + '_' + str(now_ci).replace('.', 'x')
                        columns.append(strave+suff)
                        columns.append(strstd+suff)
        
    df = pd.DataFrame(columns=columns)
    #print(df)
    cnt = len(lowlst) * len(highlst) * len(shiftlst) * len(cilst) * 2
    for i in range(num_samples):
        game_value = get_game_value()
        med = get_median_by_ywlst(game_value)
        for k in range(50):
            lst = [0.0]*cnt
            for j in range(1,num_steps+1):
                y = sample_yield_once(game_value)
                z = np.log(y+1)
                for low_bound in lowlst:
                    for high_bound in highlst:
                        for shift_value in shiftlst:
                            for ci_value in cilst:
                                now_low = round(low_bound,2)
                                now_high = round(high_bound,2)
                                now_shift = round(shift_value,2)
                                now_ci = round(ci_value,2)
                                zz = apply_bounds(z,now_low,now_high) - now_shift
                                lst.append(lst[-cnt]+zz)
                                lst.append((lst[-cnt]**now_ci+abs(zz)**now_ci)**(1.0/now_ci))
            df.loc[len(df)] = ([np.log(med+1)]+lst[cnt:])
        if i%50==49:
            print(i)
    return df