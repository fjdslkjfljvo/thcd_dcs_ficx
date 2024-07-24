import pandas as pd
import numpy as np

def zhihao_strat(state_var, prev_fills, curr_bbo_info):

    factor=[10,10,10,10]
    lags=6
    n=len(curr_bbo_info)
    # four numpy arrays
    mid, hbas, bid_sz, ask_sz = np.array(list(zip(*curr_bbo_info)))

    ######

    if not state_var:
        vtime=0
        pos = np.array([0.]*n)
        cum_impacts = np.array([0.]*n)
        last_dels = np.zeros( (lags,n) )
        last_mid = mid.copy()
    else:
        vtime = state_var[0]
        pos = state_var[1].copy()
        cum_impacts = state_var[2].copy()
        last_dels = state_var[3].copy()
        last_mid = state_var[4].copy()
        #
        vtime = vtime+1
        fill_size, avg_price = np.array(list(zip(*prev_fills)))
        pos = pos + fill_size
        this_del = np.log(mid/last_mid)
        last_dels = np.vstack( [  last_dels[1:,:], this_del   ] )
        last_mid = mid.copy()

    state_var = [vtime, pos, cum_impacts, last_dels, last_mid]
    #

    vday, vhour,vminute = (vtime//120)//24, (vtime//120)%24, vtime%120

    ######

    
    impact_decay =0.98 # impact decays by this factor after each round
    max_size = [500, 500, 1000, 150] # size submitted will be capped at max size
    max_pos = [1000, 1000, 1000, 150] # pos will be capped at max pos

    #
    arrhigh = np.zeros(n)
    arrlow = np.zeros(n)
    for i in range(n):
        arrhigh[i] = min( max_pos[i] - pos[i], max_size[i] )
        arrlow[i] = max(  -max_pos[i] - pos[i], -max_size[i]  )
        
    ######

    decision = [0.0] * n
    ######

    pred = np.array( [np.sin(2*vtime),np.sin(3*vtime),np.sin(4*vtime),np.sin(5*vtime)] )

    pred_move = ( np.exp( pred ) - 1.0 )*mid
    new_pos = np.zeros(n)
    for i in range(n):
        if pred_move[i]>0:
            new_pos_i = ( pred_move[i]-2e-4 )/1e-4
            new_pos_i*=factor[i]
            new_pos_i = max(max(new_pos_i,0),pos[i])
        else:
            new_pos_i = ( pred_move[i]+2e-4 )/1e-4
            new_pos_i*=factor[i]
            new_pos_i = min(min(new_pos_i,0),pos[i])
        new_pos[i]=new_pos_i
        decision[i]=new_pos[i]-pos[i]

    ######
    for i in range(n):
        decision[i]=max( arrlow[i], min(arrhigh[i], decision[i]) )

    ######
    return state_var, decision