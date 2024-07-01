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

def weighted_median_general(data, weights):
    # Pair each data point with its weight and sort by data point
    data_weights = sorted(zip(data, weights))
    
    total_weight = sum(weights)
    cumulative_weight = 0.0
    
    for value, weight in data_weights:
        cumulative_weight += weight
        if cumulative_weight >= total_weight / 2.0:
            return value
    
    # If for some reason the loop completes without returning, return the last value
    return data[-1]
def weighted_median(data, weights):
    max_n = 1005
    m = len(data)
    cc = [0.0]*(max_n+1)
    for i in range(m):
        if data[i]>max_n:
            cc[max_n] += weights[i]
        else:
            cc[data[i]] += weights[i]
    w = 0
    tot_weight = sum(cc) * 0.5
    for i in range(max_n+1):
        w += cc[i]
        if w>tot_weight:
            return i
def plot_series(arr):
    plt.plot(arr)
    plt.show()
def plot_dist(arr):
    plt.hist(arr, bins=np.arange(min(arr), max(arr)+1), edgecolor='black')
    plt.show()
fac_max = 155
fac=[1]
for i in range(1,fac_max):
    fac.append(fac[-1]*i)
ffac = np.array(fac).astype(float)
binom_mat = np.zeros((fac_max,fac_max))
for i in range(fac_max):
    for j in range(fac_max):
        binom_mat[i,j] = fac[i]/(fac[j]*fac[i-j])
def get_fac(n):
    #return fac[n]
    return ffac[n]
def binom(n,m):
    #return (fac[n])//(fac[m]*fac[n-m])
    #return get_fac(n)/(get_fac(m)*get_fac(n-m))
    return binom_mat[n,m]
def find_list_median(numbers):
    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    mid = n // 2
    return (sorted_numbers[mid] + sorted_numbers[~mid]) / 2
def apply_bounds(value, lower_bound, upper_bound):
    return max(lower_bound, min(value, upper_bound))
def get_game_value():
    di=100
    while di>10 or bo>12 or bo<2:
        t1=randint(2,22)
        t2=randint(2,int( (100-t1)/4 + 2 )  )
        t3=randint(1,int( (100-t1-t2)/3 + 1 )  )
        t4=randint(0,int( (100-t1-t2-t3)/2  )  )
        t5=randint(0,int( (100-t1-t2-t3-t4)  )  )
        di,bo,g,s,br = sorted([t1,t2,t3,t4,t5])
    return ([br,s,g,bo,di])
def get_yield_by_picked(picked):
    return ( (picked[0]*2+picked[1]*5+picked[2]*10)*(2**picked[4]) )  
def sample_picked_once(game_value):
    left = game_value.copy()
    picked = [0]*5
    while(sum(left)>=1 and picked[3]<2):
        now = random.choices(list(range(5)), left, k=1)[0]
        left[now] -= 1
        picked[now] += 1
    return picked
def sample_picked_once_with_time(game_value):
    left = game_value.copy()
    picked = [0]*5
    time1 = 0
    time2 = 0
    while(sum(left)>=1):
        now_time = randint(1,20)
        if picked[3]==0:
            time1 += now_time
        time2 += now_time
        now = random.choices(list(range(5)), left, k=1)[0]
        left[now] -= 1
        picked[now] += 1
        if picked[3]==2:
            break
    return picked, time1, time2
def sample_yield_once(game_value):
    picked = sample_picked_once(game_value)
    return get_yield_by_picked(picked)
def sample_median(game_value):
    lst = []
    for i in range(1000000):
        lst.append( sample_yield_once(game_value) )
    return find_list_median(lst)
def count_realization(type_value,game_value):
    type_sum = sum(type_value)
    uni_sum = sum(game_value)
    res = 1.0
    for i in range(5):
        res=res*binom(game_value[i],type_value[i])
    res=res/binom(uni_sum,type_sum)
    res=res*2.0/type_sum
    return res
def get_ywlst(game_value):
    # lack  * 2.0 * binom(game_value[3],2)
    res = []
    uni_sum = sum(game_value)
    type_sum = 2
    y = 0
    for br in range(game_value[0]+1):
        y = y + br*2
        type_sum = type_sum + br
        wbr = binom(game_value[0],br)
        for s in range(game_value[1]+1):
            y = y + s*5
            type_sum = type_sum + s
            ws = wbr * binom(game_value[1],s)
            for g in range(game_value[2]+1):
                y = y + g*10
                type_sum = type_sum + g
                wg = ws * binom(game_value[2],g)
                yy = y
                for di in range(game_value[4]+1):
                    type_sum = type_sum + di
                    now = [br,s,g,2,di]
                    w =  wg * binom(game_value[4],di) / ( binom(uni_sum,type_sum)*type_sum )
                    res.append((yy,w))
                    yy = yy*2
                    type_sum = type_sum - di
                y = y - g*10
                type_sum = type_sum -g
            y = y - s*5
            type_sum = type_sum - s
        y = y - br*2
        type_sum = type_sum - br
    return res
def get_ywlst_oldversion(game_value):
    res = []
    for br in range(game_value[0]+1):
        for s in range(game_value[1]+1):
            for g in range(game_value[2]+1):
                for di in range(game_value[4]+1):
                    now = [br,s,g,2,di]
                    w = count_realization(now,game_value)
                    y = get_yield_by_picked(now)
                    res.append((y,w))
    return res
def get_median_by_ywlst(game_value):
    ywlst = get_ywlst(game_value)
    #sorted_ywlist = sorted(ywlst)
    list_of_tuple = tuple(map(list, zip(*ywlst)))
    ylst = list_of_tuple[0]
    wlst = list_of_tuple[1]
    return weighted_median(ylst,wlst)