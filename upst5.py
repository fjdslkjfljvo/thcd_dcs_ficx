import pandas as pd
import numpy as np

def alice_strat(state_var, prev_fills, curr_bbo_info):


    #####
    
    window = 64
    entry_threshold = 1.0
    exit_threshold = 0.5
    stop_loss_threshold = 2.0
    #
    n=len(curr_bbo_info)
    # four numpy arrays
    mid, hbas, bid_sz, ask_sz = np.array(list(zip(*curr_bbo_info)))
    
    impact_decay =0.98 # impact decays by this factor after each round
    max_size = [100,100,100] # size submitted will be capped at max size
    max_pos = [200,200,200] # pos will be capped at max pos

    ######

    if not state_var:
        vtime=0
        pos = np.array([0.]*n)
        hist_prices = np.zeros( (window,n) )
        entry_price = 0.0
    else:
        vtime = state_var[0]
        pos = state_var[1].copy()
        hist_prices = state_var[2].copy()
        entry_price = state_var[3]
        #
        vtime = vtime+1
        fill_size, avg_price = np.array(list(zip(*prev_fills)))
        pos = pos + fill_size
        hist_prices = np.vstack( [  hist_prices[1:,:],  mid   ] )
        #
        
    #

    vday, vhour,vminute = (vtime//120)//24, (vtime//120)%24, vtime%120

    ######

    #
    arrhigh = np.zeros(n)
    arrlow = np.zeros(n)
    for i in range(n):
        arrhigh[i] = min( max_pos[i] - pos[i], max_size[i] )
        arrlow[i] = max(  -max_pos[i] - pos[i], -max_size[i]  )
        
    ######

    

    decision = [0.0] * n
    
    new_pos = pos.copy()

    px1=0
    px2=1
    price = hist_prices[:,px1] - hist_prices[:,px2]
    mean = np.average(price)
    std = np.std(price)
    z_val = (price[-1]-mean)/(std+1e-8)
    pos_new = min(max_pos[px1], max_pos[px2])
    
    if np.abs(pos[px1])<1e-3:
        if z_val > entry_threshold:
            new_pos[px1]=-pos_new
            new_pos[px2]=pos_new
            entry_price=price[-1]
        if z_val < -entry_threshold:
            new_pos[px1]=pos_new
            new_pos[px2]=-pos_new
            entry_price=price[-1]
    else:
        if pos[px1]>0.0:
            if entry_price - price[-1]  > stop_loss_threshold * std:
                new_pos[px1]=0.0
                new_pos[px2]=0.0
            elif z_val > -exit_threshold:
                new_pos[px1]=0.0
                new_pos[px2]=0.0
        else:
            if price[-1] - entry_price  > stop_loss_threshold * std:
                new_pos[px1]=0.0
                new_pos[px2]=0.0
            elif z_val < exit_threshold:
                new_pos[px1]=0.0
                new_pos[px2]=0.0
    #
    for i in range(n):
        decision[i]=new_pos[i]-pos[i]
    if vtime<window-1.5:
        decision = [0.0] * n
    
    ######
    ######
    for i in range(n):
        decision[i]=max( arrlow[i], min(arrhigh[i], decision[i]) )

    ######
    state_var = [vtime, pos, hist_prices, entry_price]
    return state_var, decision

