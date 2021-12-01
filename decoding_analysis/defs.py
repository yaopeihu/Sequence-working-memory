import numpy as np
from time import time
from time import sleep
import scipy.io as sio
from scipy.ndimage import gaussian_filter
from sympy.utilities.iterables import multiset_permutations
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as tmp

import multiprocessing as mp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import glob

def tn(x): return x.cpu().detach().numpy()

def trainDecoder(rnn, R, T, nl, ng, lr):
    nTr,nGo,_ = R.shape; ip = nTr//10+32; ng_ = ng+2; batch = 16; batch_ = 128; th = .5
    l=[]; opt = optim.Adam(rnn.parameters(), lr=1e-6)
    while batch < batch_:
        if opt.param_groups[0]['lr'] < lr: 
            # progressively increased the batch size  when training loss plateaus
            opt.param_groups[0]['lr']= lr; ng_ = max(ng_-1,ng); batch*=2; th = max(th/5, 0.05)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 1/2, 10, th, verbose=True)
        irands = torch.cat([torch.randperm(nTr, device=R.device) for _ in range(int(batch*ip/nTr)+1)])
        for i in range(ip):
            irand = irands[i*batch:(i+1)*batch]
            Loss = rnn.Loss(rnn(R[irand]), T[irand])
            Losss = sum([Loss[:,:,k:nGo-(ng_-1)+k] for k in range(ng_)])/ng_
            # Losss.shape = batch, sl, (nGo-ng+1), nw 
            loss = Losss.mean(-1).sort(2)[0][:,:,0].mean(0).sum() + rnn.Lwei().mean(-1).sum()/batch 
            # optimize for ng_ frames
            loss.backward(); opt.step(); opt.zero_grad()
        with torch.no_grad(): 
            Loss = rnn.Loss(rnn(R), T)
            loss = sum([Loss[:,:,k:nGo-(ng_-1)+k] for k in range(ng_)]).sort(2)[0][:,:,0] #nw
            ml0 = loss.mean(0).mean(0); ml = ml0.sort()[0][-nl:].mean().item()
        scheduler.step(ml) 
    rnn.j = tn(ml0.argsort()[nl:2*nl]) # use nl-th to 2nl-th decoders

def testDecoder(rnn, R):
    nTr, ndts, nGo, N = R.shape; Ts = []; rnn.eval() 
    with torch.no_grad():
        for i in range(ndts):
            o = torch.einsum('bsgnh,nph->bgnsp', rnn(R[:,i]), rnn.M)
            # batch, nGo, nw, sl, pl
            t = tn(o)[:, 1:-1].argmax(-1).reshape(nTr, nGo-2, rnn.nw, -1)
            # batch, nGo-2, nw, sl
            Ts.append(t[:, :, rnn.j])
    return np.stack(Ts) # ndts, batch, nGo-2, rj, sl

def getTrialWeight(T, pl, rw):
    tgW = []; tgC = []
    T = T.reshape(len(T),-1); sl = T.shape[-1]; 
    for rank in range(sl):
        tgc0 = np.array([sum(T[:,rank]==i).item() for i in range(pl)])
        tgC.append(tgc0); tgc = tgc0
        tgw = 1/(rw*tgc + tgc.mean()); tgw[tgc0==0] = 0
        tgw *= sum(tgc0)/sum(tgw*tgc0); tgW.append(tgw)
    return np.array(tgC), np.array(tgW).T

def getMarkers(fr, sl, pl, data, p=1):
    aligns = data['cue_on'].ravel(); matches = data['matches'].ravel()
    targets = data['targets']; respons = data['response']; T_ons = data['frameid']
    if sl==4: idxt = targets[:,sl-1]!=0
    else: idxt = (targets[:,sl-1]!=0)*(targets[:,sl]==0)
    T12 = T_ons[idxt,1] - T_ons[idxt,0]; T12[np.isnan(T12)] = 0
    ad = np.isnan(T_ons[idxt,sl-1]) + np.isnan(T_ons[idxt,4])
    # some adaptive trials need to be removed
    if sl==3: ad += T12<fr*(0.3+0.6)/2         
    if sl==2: ad += T12>fr*(0.6+1.2)/2
    match = matches[idxt][~ad]; target = targets[idxt,:sl][~ad]; respon = respons[idxt,:sl][~ad]
    aligns1 = aligns[idxt][~ad]; aligns2 = (aligns+T_ons[:,4])[idxt][~ad] 
    idxf = (respons[idxt,0]!=0)[~ad] # finished trials
    tgs = target[idxf]; res = respon[idxf]; mah = match[idxf]; fid = T_ons[idxt,:][~ad][idxf]
    align1 = np.round(aligns1[idxf]).astype(int) 
    align2 = np.round(aligns2[idxf]).astype(int)
    if p==1: 
        return align1, align2, tgs, res, mah, fid
    else: 
        return aligns1, aligns2, align1, align2, target, respon, match, idxf, tgs, res, mah, fid

def getDts(fid, sl, fr, nGo):
    l=[]; dts = [fid[:,0]-fr]; l.append(len(dts))
    for i in range(sl): 
        dts.extend([fid[:,i], fid[:,i]+fr*.3])
        if i<sl-1: l.append(len(dts))
    for i in range(7):  
        dts.append(np.minimum(fid[:,sl-1]+.3*fr*(i+2),fid[:,4]+.1*fr-nGo))
    l.append(len(dts))
    dts.append(fid[:,4]+.1*fr)
    for i in range(sl-1): dts.append(fid[:,6+i])
    l.append(len(dts))
    for i in range(2):  dts.append(fid[:,5+sl]+.3*fr*i)
    return dts, np.array(l)-.5    

class itemNN(nn.Module):
    def __init__(self, N, Nh, nw, sl, mw, tgW):
        super().__init__()
        self.sl = sl
        self.mw = mw
        self.nw = nw
        self.tgW = tgW
        self.j = [0]
        x = (3**.5)/2
        M0 = torch.tensor([[-.5, .5, 1, .5,-.5, -1],
                           [  x,  x, 0, -x, -x,  0]])
        self.M = nn.Parameter(torch.stack([M0.T]*nw))
        self.Wi = nn.Parameter(torch.randn([sl, nw, N, Nh])/N**.5)
        self.bi = nn.Parameter(torch.zeros([1, sl, 1, nw, Nh]))
        self.g = nn.Softmax(dim=-1)
        
    def forward(self, r):
        # r : batch, nGo, N  # Wi: sl, nw, N, Nh
        o = torch.einsum('bgi,snih->bsgnh', r, self.Wi)
        return (o + self.bi)  # batch, sl, nGo, nw, Nh 

    def Loss(self, out, t):
        # M.shape = nw, pl, Nh
        o = self.g(torch.einsum('bsgnh,nph->bsgnp', out, self.M))        
        # o.shape = batch, sl, nGo, nw, pl
        y = o[torch.arange(len(o)).view(-1,1), range(self.sl), :, :, t] 
        # y.shape = batch, sl, nGo, nw
        Loss = -y.log()*self.tgW[t, range(self.sl)].reshape(-1,self.sl,1,1)
        # t.shape = batch, sl; tgW.shape = pl, sl -> batch, sl
        return Loss
    
    def Lwei(self):
        Lw = self.Wi.pow(2).sum(-1).sum(-1) # sl, nw 
        Lm = self.mw[1]*self.M.pow(2).sum(-1).sum(-1) # nw
        return self.mw[0]*(Lw + Lm/self.sl) # sl, nw

def runShuffle(nsf, tars, tgcs, iTs, p=0):
    ss = np.random.SeedSequence(12345)
    seeds = ss.spawn(nsf)    
    def shf(seed): 
        rng = np.random.default_rng(seed)
        if p==0: 
            return extremePixel(tars, tgcs, iTs, rng)
        else:
            return extremePixel_(tars, tgcs, iTs, rng)
    with mp.Pool() as pool:
        results = pool.map(shf, seeds, 16)
    return np.array(results)

class Shuffle:
    def __init__(self, tars, tgcs, iTs, p=0, nw=10):
        self.tars = tars.astype(int); self.tgcs = tgcs; self.iTs = iTs; self.nw = nw; self.p = p
        
    def runShuffle(self, nsf):
        ss = np.random.SeedSequence(12345); seeds = ss.spawn(nsf)
        with mp.Pool() as pool: results = pool.map(self.shf, seeds)
        return np.array(results).ravel()

    def shf(self, seed):
        rng = np.random.default_rng(seed)
        if self.p==0: return self.extremePixel(rng) # Acc
        elif self.p==1: return self.extremePixelF1(rng) # F1
        else: print('Wrong flag.')

    def dummyDecoder(self, ngo, tgd, rng):        
        rands = rng.random(ngo*self.nw); tgd = np.cumsum(tgd/sum(tgd)); tgd[-1]=1
        for i in range(len(tgd)): rands[rands<tgd[i]]=i+1
        return rands-1        
        
    def extremePixel(self, rng):
        nseq, tTrain, tTest, _ = self.tgcs.shape; shf = []
        for il1 in range(tTrain):
            for il2 in range(tTest):
                nc = np.zeros(self.nw)
                for i in range(nseq):
                    tar = self.tars[self.iTs[i]:self.iTs[i+1]]
                    nc += self.dummyNc(tar, self.tgcs[i,il1,il2], rng)
                shf.append(nc/self.iTs[-1])
        return np.max(np.stack(shf,-1),-1)

    def dummyNc(self, tar, tgd, rng):
        res = np.zeros((self.nw, len(tar))); rands = rng.random(res.shape)
        tgd = tgd/sum(tgd); tgd = np.cumsum(tgd)
        for i in range(len(tgd)-1):
            res[rands>=tgd[i]]=i+1
        nc = np.sum(tar==res,-1)
        return nc

    def extremePixelF1(self, rng):
        MF1s=[]; nseq, tTrain, tTest, pl = self.tgcs.shape
        tgc0 = [sum(self.tars==i) for i in range(pl)]
        for il1 in range(tTrain):
            for il2 in range(tTest):
                Pre=[]; F1=[]
                for i in range(nseq):
                    pre = self.dummyDecoder(self.iTs[i+1]-self.iTs[i], self.tgcs[i,il1,il2], rng)
                    Pre.append(pre.reshape(self.nw, -1))
                Pre = np.concatenate(Pre, -1) # nw, nTr
                for i in range(pl):
                    tgc = tgc0[i]; tgp = (Pre==i).sum(-1)
                    nc = (Pre[:, self.tars==i]==i).sum(-1)
                    if tgc>0: F1.append(2*nc/(tgc+tgp))
                MF1s.append(np.mean(np.stack(F1),0))
        return np.max(np.stack(MF1s,-1),-1)

def getF(traces, align, dts, nGo):
    F=[]; Fp=[]; N = traces.shape[0]
    for i in range(len(align)):
        for dt in dts: F.extend([traces[:,align[i]+int(dt[i]+.5)+j] for j in range(nGo)])
    F = np.array(F).reshape(len(align),len(dts), nGo, N)
    return F
    
def getFm(F, T, pl, sl):
    seq_pool = np.array(list(multiset_permutations(range(pl))))[::np.prod(range(1,pl-sl+1)),:sl]
    tgs = tn(T); tW = np.zeros(len(tgs))
    for seq in seq_pool:
        iseq = np.where((tgs==seq).all(axis=1))[0]
        if len(iseq)>0: tW[iseq] = 1/(len(iseq)+1)
    tW /= sum(tW); tW = tW.reshape(-1,1,1);  Fm = (F.mean(-2)*tW).sum(0)
    return Fm.reshape(1,-1,1,Fm.shape[-1])

def testDecoder1(rnn, R, T, rank):
    nTr, ndts, nGo, N = R.shape; Ac = []; rnn.eval(); T = T.reshape(-1,1,1)
    with torch.no_grad():
        for i in range(ndts):
            out = rnn(R[:,i,1:-1])[:,rank] # nTr, ng, nw, Nh
            t = torch.einsum('bgnh,nph->bgnp', out, rnn.M)[:,:,rnn.j]
            ac = ((t.argmax(-1)==T)+.0).mean(); Ac.append(ac.item())
    return Ac 


def pf(a1,a2): plt.figure(figsize=(a1,a2))

def zoom(A, nz, rs=3):
    return [gaussian_filter(np.kron(a, np.ones([nz]*2)), nz/rs) for a in A]   

def pDecode(Mat, Mat0, vrange, cticks, l1, l2, nzoom, sl):
    sl0=len(Mat)//sl; pw = Mat[0].shape[1]/5; ph = len(Mat[0])/5
    Mat = zoom(Mat , nzoom); Mat0= zoom(Mat0, nzoom)
    l1 = (l1+.5)*nzoom-.5; l2 = (l2+.5)*nzoom-.5; w_r = [pw]*sl+[pw/12]; 
    pf(sum(w_r), sl0*1.0*ph); gs = gridspec.GridSpec(sl0,sl+1, width_ratios=w_r); gs.update(wspace=0.02, hspace=0.02)
    lm = Mat[0].shape[0]; ln = Mat[0].shape[1]; X, Y = np.meshgrid(np.arange(ln), np.arange(lm))
    for j in range(sl0):
        for i0 in range(sl):
            i = i0+j*sl
            plt.subplot(gs[i+j]); v_min=vrange[0]; v_max=vrange[1]
            for ll in l1:  plt.axhline(ll, c='w', lw=.75, ls='-')
            for ll in l2:  plt.axvline(ll, c='w', lw=.75, ls='-')
            plt.contour(X, Y, Mat0[i], [0,1], linestyles='--', linewidths=1.5, colors='yellow')
            plt.imshow(Mat[i], vmin=v_min, vmax=v_max, origin='lower', cmap='coolwarm')
            plt.xticks([],[]); plt.yticks([],[]); plt.box(False)
    cb = plt.colorbar(cax=plt.subplot(gs[i+j+1]), ticks=cticks); cb.outline.set_visible(False)

def pDecodeSOR(Mat, vrange, cticks, l, nzoom, sl, pR, smooth=False):
    pR[0]*=nzoom; pR[1]*=nzoom; sl0=len(Mat)//sl; pw = len(Mat[0])/3
    Mat = zoom(Mat , nzoom); Mat = [mat[pR[0]:pR[1],pR[0]:pR[1]] for mat in Mat ]; 
    l = (l+.5)*nzoom-.5; ls = [l]*sl; w_r = [pw]*sl+[pw/10]; 
    pf(sum(w_r)/2, sl0*1.02*pw/2); gs = gridspec.GridSpec(sl0,sl+1, width_ratios=w_r); gs.update(wspace=0.02, hspace=0.02)
    lm = Mat[0].shape[0]; X, Y = np.meshgrid(np.arange(lm), np.arange(lm))
    for j in range(sl0):
        for i0 in range(sl):
            i = i0+j*sl
            plt.subplot(gs[i+j]); v_min=vrange[0]; v_max=vrange[1]
            for ll in l:  plt.axhline(ll, c='C0', lw=1, ls='--'); plt.axvline(ll, c='C0', lw=1, ls='--')
            plt.imshow(Mat[i], vmin=v_min, vmax=v_max, origin='lower', cmap='Oranges', interpolation='gaussian' if smooth else 'None')
            plt.xticks([],[]); plt.yticks([],[])
    cb = plt.colorbar(cax=plt.subplot(gs[i+j+1]), ticks=cticks); cb.outline.set_visible(False)