import glob
import os
import copy
import cv2
import logging
import numpy as np
from time import time
import multiprocess as mp
import peakutils
import tifffile                  # read and save tiff

import scipy.io as sio
from scipy import interpolate
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import savgol_filter, welch
from scipy.sparse import csc_matrix

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf, params, utilities
from caiman.utils.visualization import get_contours, plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.stats import mode_robust, df_percentile
from caiman.components_evaluation import evaluate_components_CNN

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# import inspect
# get_ipython().set_next_input("".join(inspect.getsourcelines(pTraces)[0]))

def nm(x): return np.linalg.norm(x,axis=0)
def pf(a1,a2): plt.figure(figsize=(a1,a2))

def idxComp(cnm, vals):
    cnm.params.quality['SNR_lowest']  = vals[1]
    cnm.params.quality['min_SNR']     = vals[2]
    cnm.params.quality['rval_lowest'] = vals[3]
    cnm.params.quality['rval_thr']    = vals[4]
    cnm.params.quality['cnn_lowest']  = vals[5]
    cnm.params.quality['min_cnn_thr'] = vals[6]
    cnm.estimates.filter_components(np.zeros((1,512,512)), cnm.params)
    c_good  = cnm.estimates.idx_components.copy()
    bad = (cnm.estimates.SNR_comp < vals[0][0]) + (cnm.estimates.r_values < vals[0][1]) + (cnm.estimates.cnn_preds < vals[0][2]) + (cnm.estimates.SNR_comp < (vals[1] + vals[2])/2) * (cnm.estimates.r_values < (vals[3] + vals[4])/2)
    c_todo = np.setdiff1d(np.arange(cnm.estimates.nr)[~bad], c_good)
    print(len(c_todo), len(cnm.estimates.idx_components), len(cnm.estimates.idx_components_bad))
    return c_todo, c_good    

def getMask(shift, sls, dim=512):
    mask = np.zeros((dim,dim))
    mask[tuple(sls)]= 1
    xs, ys = np.array(shift).T
    x1 = np.mean(xs[xs>0])
    if ~np.isnan(x1): x1 = round(x1)
    if x1>0: mask[:x1,:] = 0
    x2 = np.mean(xs[xs<0])
    if ~np.isnan(x2): x2 = round(x2)
    if x2<0: mask[round(x2):,:] = 0
    y1 = np.mean(ys[ys>0])
    if ~np.isnan(y1): y1 = round(y1)
    if y1>0: mask[:,:round(y1)] = 0
    y2 = np.mean(ys[ys<0])
    if ~np.isnan(y2): y2 = round(y2)
    if y2<0: mask[:,round(y2):] = 0
    return mask.flatten(order='F').astype(bool)

def getSlice(x, p=[70,2], dim=512):
    # x = np.clip(x,-10,10)
    x1_= np.percentile(x, p[0])
    x1 = np.percentile(x, 100-p[1])
    x2_= np.percentile(x, p[1])
    x2 = np.percentile(x, 100-p[0])
    s1 = max(np.mean(x[(x>x1_)*(x<x1)]),0) + 1
    s2 = min(np.mean(x[(x>x2_)*(x<x2)]),0) - 1
    s = np.round([s1,s2]).astype(int)
    return slice(s[0], dim+s[1])

def deleteFiles(ftype, save=0):
    fnames = sorted(glob.glob(ftype))
    if save: fnames = fnames[:-1]
    [os.remove(fnames[i]) for i in range(len(fnames))]

def pBG(cnm):
    gnb = cnm.params.init['nb']; pf(10*gnb,9);
    for i in range(gnb):
        plt.subplot(1,gnb,i+1);
        bi = cnm.estimates.b[:,i].reshape(cnm.dims[0],cnm.dims[1],order='F'); bi = bi - np.min(bi)
        plt.imshow(bi,cmap='viridis',vmax=np.percentile(bi,99)); plt.axis('off');

def deTrend(trace0, fr, tw_sec=160, nseg_=2, keep_sec=20):
    K, T = trace0.shape
    tracef = np.apply_along_axis(gaussian_filter, 1, trace0, fr/10)
    sep = np.round(np.linspace(0,T,int(round(T/(fr*tw_sec/2)))+1)).astype('int')
    nseg = len(sep)-2; mode = np.zeros((K,nseg)); modes, traced = np.zeros((2,K,T))
    for j in range(nseg):
        trace = tracef[:,sep[j]:sep[j+2]]
        sep_ = np.round(np.linspace(0, trace.shape[1], nseg_+1)).astype('int')            
        trace_ = np.concatenate([np.sort(trace[:,sep_[k]:sep_[k+1]])[:,:int(round(fr*keep_sec))] for k in range(nseg_)], axis=1)
        mode[:,j] = mode_robust(trace_, axis=1)
        # Robust estimator of the mode of a data set using the half-sample mode.
    for i in range(K):
        f = interpolate.interp1d(sep[1:-1], mode[i,:], bounds_error=False, fill_value=(mode[i,0],mode[i,-1]))
        modes[i] = f(np.arange(T)); traced[i] = trace0[i] - modes[i]
    return modes, traced

def noiseNormalization(trace0, fr, tw_sec=240):
    nC, T = trace0.shape; traces = copy.deepcopy(trace0)
    trace_= np.zeros_like(traces); tracep = np.zeros_like(traces)
    residual = traces - savgol_filter(traces, round(fr*2)*2-1, polyorder=3, axis=-1, mode='nearest')
    sep = np.linspace(0,T,min(int(round(T/fr/tw_sec)), int(round(T/1024)))).astype('int')
    nseg = len(sep)-2; sns = np.zeros((nC,nseg))
    for i in range(nC):
        for j in range(nseg):
            sns[i,j] = np.std(residual[i,sep[j]:sep[j+2]])
        f = interpolate.interp1d(sep[1:-1], sns[i], kind='quadratic', bounds_error=False, fill_value=(sns[i,0],sns[i,-1]))
        trace_[i] = traces[i]/f(np.arange(T))
        tracep[i] = gaussian_filter(trace_[i],fr/4)
    return trace_, tracep

def deTrend_(cnm, n0, tw_sec=160, nseg_=2, keep_sec=20): # for one component
    trace0 = cnm.estimates.C[n0] + cnm.estimates.YrA[n0]; fr = cnm.params.data['fr']
    tracef = gaussian_filter(trace0, fr/10); T = len(trace0)
    sep = np.round(np.linspace(0,T,int(round(T/(fr*tw_sec/2)))+1)).astype('int')
    nseg = len(sep)-2; mode = np.zeros(nseg)
    for j in range(nseg):
        trace = tracef[sep[j]:sep[j+2]]
        sep_ = np.round(np.linspace(0, len(trace), nseg_+1)).astype('int')            
        trace_ = np.concatenate([np.sort(trace[sep_[k]:sep_[k+1]])[:int(round(fr*keep_sec))] for k in range(nseg_)])
        mode[j] = mode_robust(trace_)
    f = interpolate.interp1d(sep[1:-1], mode, bounds_error=False, fill_value=(mode[0],mode[-1]))
    return trace0 - f(np.arange(T))

def noiseNormalization_(trace, fr):
    T = len(trace)                  # for one component
    residual = trace - savgol_filter(trace, round(fr*2)*2-1, polyorder=3, mode='nearest')
    sep = np.linspace(0,T,min(int(round(T/fr/240)), int(round(T/1024)))).astype('int')
    nseg = len(sep)-2; sns = np.zeros(nseg)
    for j in range(nseg): sns[j] = np.std(residual[sep[j]:sep[j+2]])
    f = interpolate.interp1d(sep[1:-1], sns, kind='quadratic', bounds_error=False, fill_value=(sns[0],sns[-1]))
    return trace/f(np.arange(T))

def thresComp(cnm, thresA):
    A_thr = []
    for i in range(cnm.estimates.A.shape[1]):
        Ai = cnm.estimates.A[:,i].toarray().reshape(cnm.dims[0],cnm.dims[1])
        A_ = median_filter(Ai, (3,3))
        A_thr.append((A_ > np.max(A_)*thresA).astype(int))    
    return csc_matrix(np.array(A_thr).reshape(-1,cnm.dims[0]*cnm.dims[1]).T)
    
def pComp(cnm, Ct, Cn, n0, events, I, p=0):
    if I==0: idx = np.arange(cnm.estimates.A.shape[1])
    if I==1: idx = cnm.estimates.idx_components
    if I==2: idx = cnm.estimates.idx_components_bad
    d0 = cnm.params.init['gSig'][0]; fr = cnm.params.data['fr']
    na = idx[n0]; da = int(round(5*d0)); d1, d2 = cnm.dims;   
    a1, a2 = np.round(cnm.estimates.coordinates[na]['CoM']).astype(int)
    print(n0, '-', na, '[',a1, a2,']', ' %.2f, %.2f, %.4f' 
          %(cnm.estimates.r_values[na],cnm.estimates.SNR_comp[na],cnm.estimates.cnn_preds[na]))
    nGo= np.round(cnm.estimates.f.shape[-1]/700/fr).astype(int)
    tR = np.arange(cnm.estimates.f.shape[-1]//nGo); T = len(tR)
    traced = deTrend_(cnm, na); trace = noiseNormalization_(traced, fr)
    trace_p = np.clip(gaussian_filter(trace, fr/4), 0, None)
    peaks = peakutils.indexes(trace_p, thres=.45, min_dist=3*fr)
    if p<2:
        pf(48,3.5); plt.subplot(1,12,1)
        Ai = cnm.estimates.A[:,na].toarray().reshape(d1,d2,order='F')        
        ai = Ai[max(0,a1-da):a1+da,max(0,a2-da):a2+da]
        plt.imshow(ai,clim=[1e-9,np.percentile(ai,99)]); plt.axis(False)
        plt.subplot(1,12,2)
        c1, c2, c3 = Ct.shape; ra = c1/d1
        r1_= int(round(ra*(a1-da))); r2_= int(round(ra*(a2-da)))
        r1 = int(round(ra*(a1+da))); r2 = int(round(ra*(a2+da)))
        plt.imshow(Ct[max(0,r1_):r1,max(0,r2_):r2,:]); plt.axis(False)
        plt.subplot(1,12,3)
        cn = Cn[max(0,a1-da):a1+da,max(0,a2-da):a2+da]
        plt.imshow(cn, clim=[np.percentile(Cn,5), np.percentile(Cn,99)]); plt.axis(False)
        plt.subplot(1,12,(4,6)); hp = []    
        for i in range(len(peaks)): 
            try: hp.append(peaks[i]-events[events<(peaks[i]+fr+1)][-1])
            except: pass
        bs = np.linspace(-fr-2,fr*8,50); plt.hist(hp,bins=bs,ec='w')
        plt.text(bs[0]+fr/8, plt.ylim()[1]*.9, na, fontsize=18)
        plt.axvline(0,lw=.8,c='C1'); plt.xlim(bs[0],bs[-1])
        plt.subplot(1,12,(7,9));
        ff, Pxx = welch(traced); plt.plot(ff,Pxx,'.-')
        plt.ylim(bottom=0); plt.yticks([],[]); plt.xlim(-.01,.51); plt.show()
    if p>0:
        nGo_ = (nGo+1)//2; n=0
        if p==2: n += sum((events>tR[0])*(events<=tR[-1])); tR+=T; nGo_=nGo//2        
        pf(48,nGo_*2.5); y_m = np.percentile(trace,99)
        for j in range(nGo_):
            plt.subplot(nGo_,1,j+1); plt.plot(trace[tR],'|',ms=3,c='C1')
            event_ = events[events>tR[0]]-tR[0]; event_ = event_[event_<=T]
            peaks_ =  peaks[ peaks>tR[0]]-tR[0]; peaks_ = peaks_[peaks_<=T]
            for i in range(len(peaks_)): 
                plt.plot(peaks_[i], -y_m/14, '^', ms=4, c='k')
            for i in range(len(event_)):
                plt.axvline(event_[i], lw=.8, c='gray'); n+=1 
            plt.plot(trace_p[tR],lw=2,c='C0'); plt.axhline(0,lw=0.5,ls=':',c='k')
            plt.xlim(0,T); plt.ylim(-y_m/9,y_m*1.5); plt.xticks([],[]); tR+=T
            n += sum((events>tR[0])*(events<=tR[-1])); tR+=T
        plt.xticks(np.arange(1,T//1000)*1000, np.arange(1,T//1000)*1000); plt.show()

def pCompBG(cnm, events, n0): # plot background component
    fr = cnm.params.data['fr']
    nGo= np.round(cnm.estimates.f.shape[-1]/800/fr).astype(int)
    tR = np.arange(cnm.estimates.f.shape[-1]//nGo); T = len(tR)
    trace = cnm.estimates.f[n0]/10000
    trace_p = gaussian_filter(trace, fr/4)
    pf(48,nGo*2.5);
    gs = gridspec.GridSpec(nGo,1); gs.update(hspace=0.05)
    for j in range(nGo):
        plt.subplot(gs[j])
        plt.plot(trace[tR],'|',ms=3,c='C1')
        plt.plot(trace_p[tR],  lw=2,c='C0')
        event_ = events[(events-tR[0])>0]-tR[0]; event_ = event_[event_<T]
        for i in range(len(event_)): 
            plt.axvline(event_[i],lw=.5,c='C2')
        plt.axhline(0,lw=0.5,ls=':'); plt.xlim(0,T)
        tR = tR + T
        if j<(nGo-1): plt.xticks([],[])

def pCompCont(cnm, Cn, idxs=[]):
    pf(40,19); d1, d2 = cnm.dims; ip=0 
    gs = gridspec.GridSpec(1,2); gs.update(wspace=0.01)
    if len(idxs)==0:
        idxs = [cnm.estimates.idx_components, cnm.estimates.idx_components_bad]
    for idx in idxs:
        plt.subplot(gs[ip]); ip+=1; patches = []
        if len(idx)==0: plt.axis(False); continue
        coor = [cnm.estimates.coordinates[i]['coordinates'] for i in idx]
        com = np.array([cnm.estimates.coordinates[i]['CoM'] for i in idx])
        plt.imshow(Cn, cmap='viridis', vmin=0, vmax=np.percentile(Cn,99))
        plt.plot(com[:,1], com[:,0], 'ko', ms=2)
        c1 = [np.clip(cor[1:-1, 0], 0, d2) for cor in coor]
        c2 = [np.clip(cor[1:-1, 1], 0, d1) for cor in coor]
        for i in range(len(c1)):
            patches.append(Polygon(np.array([c1[i],c2[i]]).T))
            plt.text(com[i,1], com[i,0], str(idx[i]), c='w', fontsize=14)
        plt.gca().add_collection(PatchCollection(patches,ec='r',lw=1.5,fc='None'))
    
def pMetric(cnm):
    pf(30,4); gs = gridspec.GridSpec(1,3); gs.update(wspace=0.1)
    plt.subplot(gs[0]); plt.hist(cnm.estimates.SNR_comp , bins = np.arange(0,50,0.3), ec='w') 
    plt.xticks(np.arange(0,16)); plt.xlim(0,16); plt.title('SNR_ratio', fontsize=18)
    plt.subplot(gs[1]); plt.hist(cnm.estimates.r_values , bins = np.linspace(0,1,50), ec='w')
    plt.xticks(np.arange(0,1.1,.1)); plt.xlim(-.03,1.03); plt.title('r_value', fontsize=18)
    plt.subplot(gs[2]); plt.hist(cnm.estimates.cnn_preds, bins = np.linspace(0,1,50), ec='w')
    plt.xticks(np.arange(0,1.1,.1)); plt.xlim(-.03,1.03); plt.title('cnn_preds', fontsize=18)
    
def getCt(cnm, Cn, filepath, idxs=[]):
    d1, d2 = cnm.dims; pf(20,20); lss = ['-','--']   
    if len(idxs)==0: idxs = [cnm.estimates.idx_components, cnm.estimates.idx_components_bad]
    plt.imshow(Cn, cmap='viridis', vmin=0, vmax=np.percentile(Cn,99)); k=0
    for (k,idx) in enumerate(idxs):
        coor = [cnm.estimates.coordinates[i]['coordinates'] for i in idx]
        com = np.array([cnm.estimates.coordinates[i]['CoM'] for i in idx])
        patches = []; plt.plot(com[:,1], com[:,0], 'ko', ms=2)
        c1 = [np.clip(cor[1:-1, 0], 0, d2) for cor in coor]
        c2 = [np.clip(cor[1:-1, 1], 0, d1) for cor in coor]
        for i in range(len(c1)): 
            patches.append(Polygon(np.array([c1[i],c2[i]]).T))
            plt.text(com[i,1], com[i,0], str(idx[i]), c='w', fontsize=12)
        plt.gca().add_collection(PatchCollection(patches,ec='r',lw=1,ls=lss[k],fc='None'))
    plt.axis(False)
    plt.savefig(filepath + 'Ct.png', bbox_inches='tight', pad_inches=0, dpi=120); plt.close()
    return mpimg.imread(filepath+'Ct.png')

def saveComp(cnm, c_save):
    cnm.estimates.idx_components_bad = np.setdiff1d(cnm.estimates.idx_components_bad, c_save)
    cnm.estimates.idx_components = np.setdiff1d(np.arange(cnm.estimates.nr), cnm.estimates.idx_components_bad)
    idxc = cnm.estimates.idx_components; idxb = cnm.estimates.idx_components_bad
    print(len(idxc),len(idxb))
    return cnm, idxc, idxb

def evaComp(cnm, Yr, breaks, sls=[], use_CNN=0):
    _, traced = deTrend(cnm.estimates.C + cnm.estimates.YrA, cnm.params.data['fr'])
    cnm.estimates.SNR_comp = getSNR(traced, cnm.params.data['decay_time']>=1)
    Px = getPx(cnm); rval_t, rval_s = getRvals(cnm, Yr, Px, Px, breaks)
    cnm.estimates.r_values = np.maximum(rval_t, rval_s)
    if use_CNN:
        predictions = evaluate_components_CNN(cnm.estimates.A, cnm.dims, cnm.params.init['gSig'])
        cnm.estimates.cnn_preds = predictions[0][:,1]
    if len(sls)>0:
        mask = np.zeros((512,512)); mask[tuple(sls)]= 1; mask = mask.flatten('F').astype(bool)
        for n0 in range(cnm.estimates.nr): # set value for edge component
            A = cnm.estimates.A[:,n0].toarray(); a = sum(A[mask])/sum(A)
            if a < .66: cnm.estimates.SNR_comp [n0] = 0      
            if a < .88: cnm.estimates.cnn_preds[n0] = 0.5

def getPx(cnm, th_a=.1):
    Px = []
    for i in range(cnm.estimates.A.shape[1]):
        a = cnm.estimates.A[:,i].toarray().reshape(cnm.dims[0],cnm.dims[1])
        a_ = median_filter(a, (3,3)); a = a_.reshape(-1)
        Px.append(set(np.where(a > max(a)*th_a)[0]))
    return Px

def getRvals(cnm, Yr, Px1, Px2, events):
    # a custom function to calculate r_value, modified based on the original method in caiman
    fr = cnm.params.data['fr']; tA = int(round(5*fr/6)); tB = int(round(-fr/6))
    A = cnm.estimates.A; C = cnm.estimates.C; K, T = np.shape(C)
    nA = np.sqrt(A.power(2).sum(0)); AA = (A.T*A).toarray()/np.outer(nA,nA.T)-np.eye(K)
    Peaks = []; nPeaks =[]; LOC=[]; er=0; n_s = 10
    nP_min = max(n_s*2,int(round(T/fr/800))); nP_max = max(nP_min, int(round(T/fr/100)))
    for i in range(K):    
        Cp = gaussian_filter(C[i,:],fr/4); I=[]; th_P=.6
        while len(I) < nP_min:
            I = peakutils.indexes(Cp, thres=th_P, min_dist=tA-tB-1); th_P *= .9            
            dp = np.array([events[np.argmin(abs(events-I[j]))] - I[j] for j in range(len(I))])
            I = I[(dp >= tA) + (dp <= tB)]
        Peaks.append(I[np.argsort(Cp[I])][::-1][:nP_max]); nPeaks.append(len(Peaks[-1]))
        interval = np.kron(I,np.ones(tA-tB)) + np.kron(np.ones(len(I)),np.arange(tB, tA))
        LOC.append(set(interval.astype(int)))
        if len(interval)!=len(set(interval.astype(int))): er+=1 
    if er>0: print('Something went wrong')
    rval_t, rval_s = np.zeros((2,K))
    for i in range(K):     
        I = Peaks[i]; nP = nPeaks[i]; ipeak = 0; loc = set([]); loc_ = set([])
        for j in np.where(AA[:,i]>.025)[0]: loc_.update(LOC[j])
        while len(loc)<n_s*(tA-tB) and ipeak<nP:
            interval = I[ipeak] + np.arange(tB,tA); ipeak+=1
            loc.update(set(interval.astype(int)) - loc_)
        a = A[:,i].toarray().reshape(-1)
        if ipeak != nP:
            rval_t[i] = pearsonr(np.mean(Yr[list(Px1[i])][:,list(loc)],1), a[list(Px1[i])])[0]
        px = Px2[i].copy();
        for j in np.where(AA[:,i]>.01)[0]: px -= Px2[j]
        if len(px) > len(Px2[i])*2/3 or len(px)>70:
            interval = np.kron(I,np.ones(tA-tB)) + np.kron(np.ones(len(I)),np.arange(tB, tA))
            loc = interval.astype(int)[:n_s*(tA-tB)]    
            rval_s[i] = pearsonr(np.mean(Yr[list(px)][:,list(loc)],1), a[list(px)])[0]
    return rval_t, rval_s

def getSNR(traced, p=True):
    # based on the ratio between low and high domain of power spectral density
    # the output does not necessarily reflect SNR in conventional sense 
    # but served as a surrogate to replace the value of SNR_comp
    snr=[]
    if p: pR=range(2,5); wei=np.array([1.5,1.5,1])/4
    else: pR=range(1,4); wei=np.array([1.0,2.0,1])/4
    for n0 in range(len(traced)):
        ff, Pxx = welch(traced[n0]) # default detrend
        snr.append((Pxx[pR]@wei)/np.median(Pxx[(ff>.14)*(ff<.36)]))
    return np.array(snr)

def getTW(cor, cor_, cor_sm_, th_1, aligns, ses_stt, lw, lw_):
    T_raw = np.where(cor_sm_ > th_1)[0]; TW = [[0,0]]; TW_ = []
    idx_stt = [0] + list(np.where(np.diff(T_raw)>1)[0]+1) + [len(T_raw)]
    for i in range(len(idx_stt)-1):
        idx1 = T_raw[idx_stt[i]]; idx2 = T_raw[idx_stt[i+1]-1]; TW_.append([idx1, idx2])
        try: it1 = np.where((idx1-aligns)<=0)[0][0]; it2 = np.where((aligns-idx2-lw_)<=0)[0][-1]        
        except: continue # the tw is in front of any trial
        for ii in range(5):
            if it1==0: idx1 = aligns[0] - lw_; break
            it1 -= 1; t1 = aligns[it1]; t2 = aligns[it1+1]; t = min(t1+lw, t2-lw_)
            t_w = range(int(t1+.5), int(t+.5))
            if min(cor_[int(t1+.5):int(t2+.5)])>0 and (t2-t1<4*lw) and np.mean(cor[t_w])>(th_1*(1-.01*(ii<2))): idx1 = t1 - lw_
            else: idx1 = t2 - lw_; break
        for ii in range(5):
            t = aligns[it2]+lw if it2==len(aligns)-1 else min(aligns[it2]+lw, aligns[it2+1]-lw_)
            t_w = range(int(aligns[it2]+.5), int(t+.5))
            if np.mean(cor[t_w])>(th_1*(1-.01*(ii<2))) or (min(cor_[t_w])>0 and (idx2-aligns[it2]>lw)): 
                idx2 = t+lw_; it2 += 1
            else: idx2 = min(aligns[it2-1]+lw+lw_, aligns[it2]-lw_); break                 
            if it2 == len(aligns): break
            if (aligns[it2] - aligns[it2-1]) >= 4*lw: break
            if min(cor_[int(aligns[it2-1]+.5):int(aligns[it2]+.5)])==0: break        
        if idx1 <= TW[-1][-1]: TW[-1][-1] = idx2 
        else: TW.append([idx1,idx2])
    TW = np.array(TW[1:]); TW = np.delete(TW, np.where(np.diff(TW,1) < 1.25*lw)[0], axis=0)  # delete some negative tw from empty slice

    Tdiss=[] # connect small tws
    for i in range(len(TW)-1): 
        if (TW[i+1][0] - TW[i][-1]) < lw:
            if sum((aligns>=TW[i][-1])*(aligns<TW[i+1][0]))==0:
                if ((TW[i][-1]<ses_stt)==(TW[i+1][0]<ses_stt)).all(): 
                    TW[i+1][0] = TW[i][0]; Tdiss.append(i)
    TW = np.delete(TW, Tdiss, axis=0)

    Tdiss=[] # delete isolated tws
    Tgap = [1e5] + list(TW[1:,0] - TW[:-1,1]) + [20*lw]
    for i in range(len(Tgap)-2):
        if Tgap[i]>40*lw and Tgap[i+2]>40*lw:
            if TW[i+1,1] - TW[i,0]<16*lw: Tdiss.extend([i,i+1])
    TW = np.delete(TW, Tdiss, axis=0)
    return TW, TW_

