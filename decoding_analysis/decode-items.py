# code for leave-one-trial-out (loto) or leave-one-sequence-out (loso) decoding

from defs import *

def handler(P): 
    # handler function for each parallel process
    print(P['p']); sl = P['sl']; pl = P['pl']; N = P['N']
    
    # read data and get trial info from mat file
    data = sio.loadmat('/mnt/sdb1/mats/more_'+P['datadate']+'.mat'); fr = data['fr'].item()
    align, _, tgs, _, mah, fid = getMarkers(fr, sl, pl, data)
        
    # perform temporal smoothing with half gaussian kernel
    sig = fr/5; ker = np.exp(-np.arange(int(4*sig+.5))**2/2/sig**2); ker = ker/sum(ker)
    traces = np.apply_along_axis(np.convolve, 1, np.pad(data['F_'], ((0,0), (len(ker)-1,0)), 'edge'), ker, 'valid')
    
    # get resampled time stamps
    nGo = int(.5*fr+.5)+1 # number of frames in each window
    dts, _ = getDts(fid, sl, fr, nGo-1); ndts = len(dts); F = []
    for i in range(len(align)):
        for dt in dts: F.extend([traces[:,align[i]+int(dt[i]+.5)+k] for k in range(nGo)])
    
    # assemble data into shape: nTrial, number of time windows, frames in each window, neurons
    F = np.array(F).reshape(-1, ndts, nGo, N) 
    Fm = (F.mean(-2)*P['tW']).sum(0).reshape(1,-1,1,N); F =  F - Fm # demean
    F_mah = F[mah==1]; T_mah = tgs[mah==1]-1 # data and target for correct trials
    return TaT(F_mah, T_mah, F, P)

def TaT(F, T, Fp, P): # Train and Test
    Ts=[]; tgCs=[]; 
    P['dv'] = torch.device('cuda:'+str(P['p']%4))
    iTest = P['iTest']; iTrain = np.setdiff1d(np.arange(len(T)), iTest)
    T = torch.tensor(T).long().to(P['dv'])
    F = torch.tensor(F).float().to(P['dv'])
    ndts = F.shape[-3]
    
    # testing data for permutation test
    Fp = torch.tensor(Fp).float().to(P['dv'])
    # trial weight for balancing trials of different items
    tgw = torch.tensor(getTrialWeight(T[iTrain], P['pl'], P['rw'])[1]).to(P['dv'])
    
    for ilook in range(ndts):
        rnn = itemNN(P['N'], P['Nh'], P['nw'], P['sl'], P['mw'], tgw).to(P['dv'])
        trainDecoder(rnn, F[iTrain, ilook], T[iTrain], P['nl'], P['ng'], P['lr'])
        
        # get decoded items for left-out trial
        Ts.append(testDecoder(rnn, F[iTest]))
        
        # get decoded item distribution for testing data
        tp = testDecoder(rnn, Fp).reshape(ndts, -1, P['sl'])
        tgCs.append(np.stack([tp==i for i in range(P['pl'])], -1).sum(1))

    print(P['p']) # finishing
    return np.array(Ts), np.array(tgCs)


if __name__ == '__main__':
    tmp.set_start_method('spawn', force=True) 
    for method in ['loto', 'loso']:
        datadate = '20190903'
        sl = 3 # sequence length
        pl = 6 # number of items
        P = {'Nh':2,  # hidden layer units
             'rw':3,  # weighting coefficient for imbalance classification
             'nw':50, # total number of decoders to train
             'nl':10, # number of decoders to use for analysis
             'ng':4,  # number of frames to optimize for in each time window 
             'mw':[20, .1], # regularization strength
             'lr':1e-3, # learning rate in gradient descent
             'pl':pl, 'sl':sl, 'datadate':datadate}
        data = sio.loadmat('/mnt/sdb1/mats/more_'+datadate+'.mat'); fr = data['fr'].item(); P['N'] = len(data['F_'])
        # get all possible non-repeating sequences for length-sl
        seq_pool = np.array(list(multiset_permutations(range(pl))))[::np.prod(range(1,pl-sl+1)),:sl]
        _, _, tgs, _, mah, _ = getMarkers(fr, sl, pl, data); tgs -=1; tar = tgs[mah==1]
        
        # get sequence based trial weight for demean  
        tW = np.zeros(len(tgs))
        for seq in seq_pool:
            iseq = np.where((tgs==seq).all(axis=1))[0]
            if len(iseq)>0: tW[iseq] = 1/(len(iseq)+1)
        P['tW'] = (tW/sum(tW)).reshape(-1,1,1) 

        Ps=[]; p=0
        if 'loso' in method:
            for seq in seq_pool: 
                iTest = np.where((tar==seq).all(axis=1))[0]; nt = len(iTest)
                if nt>0: 
                    P_=P.copy(); P_['iTest']=iTest; P_['p']=p; p+=1; Ps.append(P_)
        else: # 'loto'
            for i in range(len(tar)): 
                P_ = P.copy(); P_['iTest'] = [i]; P_['p']=p; p+=1; Ps.append(P_)

        Ts=[]; tgCs=[]; nps = 8 # number of parallel processes
        for k in range(int(np.ceil(len(Ps)/nps))):
            Ps_ = Ps[k*nps:(k+1)*nps]
            with tmp.Pool(nps) as pool: 
                results = pool.map(handler, Ps_, 1)
                for result in results: 
                    Ts.append(result[0]); tgCs.append(result[1])
            sleep(8)
        
        print(datadate, method)
        Ts = np.concatenate(Ts,2); print(Ts.shape) # ndts, ndts, nTrail, ng, nw, sl
        tgCs = np.array(tgCs); print(tgCs.shape)  # nSeq/nTrial, ndts, ndts, sl, pl
        ll = '_'; filepath = '/mnt/sdb1/TwoPhoton/'+datadate+'/'
        lstring = datadate +'_N'+ str(P['N']) +'_H'+ str(P['Nh']) +'_rw'+ str(P['rw']) +'_mw'+ str(P['mw']) +'-'+ str(P['lr'])[1:]
        np.save(filepath + method +ll+ 'lstring'+str(sl), lstring)
        np.save(filepath + method +ll+ 'tgCs'+str(sl), tgCs)
        np.save(filepath + method +ll+ 'Ts'+str(sl), Ts)
        
            
