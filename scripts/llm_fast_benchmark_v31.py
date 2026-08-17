"""
=============================================================================
🚀 FAST COMPREHENSIVE LLM OPTIMIZER BENCHMARK v3.1
   Muon + Prodigy + SF-AdamW + D-Adapt - Quick Test Mode
=============================================================================

Optimized for speed: 3 trials per optimizer, smaller model.
Tests all 10 optimizers with hyperparameter search on Modal GPU.
=============================================================================
"""

import time
import json
import os
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import modal

app = modal.App("llm-optimizer-benchmark-v31")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "numpy>=1.26.0",
        "optuna>=3.4.0",
    )
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

@dataclass
class FastConfig:
    num_layers=4; hidden_dim=256; num_heads=4; mlp_ratio=4
    vocab_size=4096; max_seq_len=128; batch_size=16; num_epochs=2
    warmup_steps=20; max_grad_norm=1.0
    lr_range=(1e-5, 1e-3); wd_range=(0.0, 0.3); n_trials=5

ALL_OPTIMIZERS = [
    ("adamw", "AdamW", "Baseline"),
    ("lion", "Lion", "Google 2023"),
    ("sophia", "Sophia", "Second-order 2024"),
    ("adafactor", "Adafactor", "Memory-efficient"),
    ("adan", "Adan", "Nesterov 2022"),
    ("radam", "RAdam", "Rectified 2020"),
    ("muon", "Muon", "NEW: SOTA Dec 2024!"),
    ("prodigy", "Prodigy", "NEW: Parameter-free"),
    ("sf_adamw", "SF-AdamW", "NEW: Schedule-free"),
    ("d_adam", "D-Adam", "NEW: LR-free Meta"),
]


@app.function(image=image, gpu="A10G", timeout=1800, memory=24576)
def run_fast_benchmark():
    import torch; import torch.nn as nn; import torch.nn.functional as F
    import numpy as np; import optuna
    from optuna.pruners import MedianPruner; from optuna.samplers import TPESampler
    
    logger.info("="*60); logger.info("FAST OPTIMIZER BENCHMARK v3.1"); logger.info("="*60)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if device=='cuda' else 'N/A'}")
    cfg = FastConfig()
    
    # ===== OPTIMIZERS =====
    class Lion(torch.optim.Optimizer):
        def __init__(s,params,lr=1e-4,betas=(0.9,0.99),wd=0):
            super().__init__(params,dict(lr=lr,betas=betas,weight_decay=wd))
        @torch.no_grad()
        def step(s,closure=None):
            for g in s.param_groups:
                lr,b1,b2,wd=g['lr'],g['betas'][0],g['betas'][1],g['weight_decay']
                for p in g['params']:
                    if p.grad is None: continue
                    st=s.state[p]
                    if 'e' not in st: st['e']=torch.zeros_like(p)
                    st['e'].mul_(b1).add_(p.grad,alpha=1-b1)
                    if wd: p.data.mul_(1-lr*wd)
                    p.data.add_(torch.sign(st['e']),alpha=-lr)
            return closure() if closure else None
    
    class Sophia(torch.optim.Optimizer):
        def __init__(s,params,lr=1e-4,betas=(0.965,0.99),rho=0.04,wd=0.1,k=10):
            super().__init__(params,dict(lr=lr,betas=betas,rho=rho,weight_decay=wd,k=k))
            s.c=0
        @torch.no_grad()
        def step(s,closure=None):
            s.c+=1
            for g in s.param_groups:
                lr,b1,b2,rho,wd,k=g['lr'],g['betas'][0],g['betas'][1],g['rho'],g['weight_decay'],g['k']
                for p in g['params']:
                    if p.grad is None: continue
                    st=s.state[p]
                    if len(st)==0: st['m']=torch.zeros_like(p);st['h']=torch.zeros_like(p)
                    m,h=st['m'],st['h'];m.mul_(b1).add_(p.grad,alpha=1-b1)
                    if s.c%k==0: h.mul_(b2).addcmul_(p.grad,p.grad,value=1-b2)
                    if wd: p.data.mul_(1-lr*wd)
                    p.data.add_(m/(h.clamp(max=rho).sqrt()+1e-15),alpha=-lr)
            return closure() if closure else None
    
    class Muon(torch.optim.Optimizer):
        def __init__(s,params,lr=1e-3,mom=0.95,wd=0.01):
            super().__init__(params,dict(lr=lr,momentum=mom,weight_decay=wd))
        @torch.no_grad()
        def step(s,closure=None):
            for g in s.param_groups:
                lr,mom,wd=g['lr'],g['momentum'],g['weight_decay']
                for p in g['params']:
                    if p.grad is None: continue
                    if p.dim()>=2:
                        st=s.state[p]
                        if 'mu' not in st: st['mu']=torch.zeros_like(p)
                        st['mu'].mul_(mom).add_(p.grad)
                        if wd: p.data.add_(p.data,alpha=-lr*wd)
                        try:
                            if p.data.numel()>500:
                                U,S,V=torch.linalg.svd(p.data.float(),full_matrices=False)
                                p.data.add_((st['mu']-0.01*(U@torch.diag(S)@V)).to(p.data.dtype),alpha=-lr*0.01)
                            else: p.data.add_(st['mu'],alpha=-lr*0.01)
                        except: p.data.add_(st['mu'],alpha=-lr*0.01)
                    else:
                        st=s.state[p]
                        if 'b' not in st: st['b']=torch.zeros_like(p)
                        st['b'].mul_(mom).add_(p.grad)
                        if wd: p.data.add_(p.data,alpha=-lr*wd)
                        p.data.add_(st['b'],alpha=-lr)
            return closure() if closure else None
    
    class Prodigy(torch.optim.Optimizer):
        def __init__(s,params,lr=1.0,betas=(0.9,0.999),wd=0,d_coef=1.0):
            super().__init__(params,dict(lr=lr,betas=betas,weight_decay=wd,d_coef=d_coef));s.step_count=0
        @torch.no_grad()
        def step(s,closure=None):
            s.step_count+=1
            for g in s.param_groups:
                dc=g['d_coef'];d=max((p.grad.abs().max().item()**2 if p.grad is not None else 0) for p in g['params']);d=max(d,1e-12)
                for p in g['params']:
                    if p.grad is None: continue
                    gr=p.grad.data;st=s.state[p]
                    if len(st)==0: st['ea']=torch.zeros_like(p);st['esq']=torch.zeros_like(p)
                    ea,esq=st['ea'],st['esq'];b1,b2=g['betas']
                    ea.mul_(b1).add_(gr,alpha=1-b1);esq.mul_(b2).addcmul_(gr,gr,value=1-b2)
                    bc1,bc2=1-b1**s.step_count,1-b2**s.step_count
                    dn=d/(p.data.norm().item()+1e-8);alr=g['lr']/(1+dc*dn)
                    if s.step_count<50: alr*=min(1.,s.step_count/50)
                    den=(esq/bc2).sqrt().add(1e-8);upd=(ea/bc1)/den
                    if g['weight_decay']: p.data.add_(p.data,alpha=-alr*g['weight_decay'])
                    p.data.add_(upd,alpha=-alr)
            return closure() if closure else None
    
    class SFAdamW(torch.optim.Optimizer):
        def __init__(s,params,lr=1e-3,betas=(0.9,0.999),wd=0.01,r=0.0):
            super().__init__(params,dict(lr=lr,betas=betas,weight_decay=wd,r=r));s.sc=0
        @torch.no_grad()
        def step(s,closure=None):
            s.sc+=1
            for g in s.param_groups:
                lr,(b1,b2),wd,r=g['lr'],g['betas'],g['weight_decay'],g['r']
                for p in g['params']:
                    if p.grad is None: continue
                    gr=p.grad.data;st=s.state[p]
                    if len(st)==0: st['step']=0;st['ea']=torch.zeros_like(p);st['esq']=torch.zeros_like(p);st['z']=p.clone()
                    ea,esq,z=st['ea'],st['esq'],st['z'];st['step']+=1;step=st['step']
                    ea.mul_(b1).add_(gr,alpha=1-b1);esq.mul_(b2).addcmul(gr,gr,value=1-b2)
                    bc1,bc2=1-b1**step,1-bl2**step if 'bl2' in dir() else 1-b2**step
                    bc2=1-b2**step;den=(esq/bc2).sqrt().add(1e-8);upd=(ea/bc1)/den
                    if wd: p.data.mul_(1-lr*wd**(2 if True else 1))
                    p.data.add_(upd,alpha=-lr)
                    bl=min(r+(1-r)*step/500,1.) if r<1 else 1.
                    z.mul_(bl).add_(p.data,alpha=1-bl)
                    if step>10: p.data.copy_(z)
            return closure() if closure else None
    
    class DAdam(torch.optim.Optimizer):
        def __init__(s,params,lr=1.0,betas=(0.9,0.999),wd=0,d0=1e-6,fgr=0.4):
            super().__init__(params,dict(lr=lr,betas=betas,weight_decay=wd,d0=d0,fs_growth_rate=fgr))
            s._d=d0;s._fs=0;s._sc=0
        @torch.no_grad()
        def step(s,closure=None):
            s._sc+=1;gn2=sum(p.grad.data.norm().item()**2 for g in s.param_groups for p in g['params'] if p.grad is not None)
            for g in s.param_groups:
                d0,fgr=g['d0'],g['fs_growth_rate']
                for p in g['params']:
                    if p.grad is None: continue
                    gr=p.grad.data;st=s.state[p]
                    if len(st)==0: st['ea']=torch.zeros_like(p);st['esq']=torch.zeros_like(p);st['step']=0
                    ea,esq=st['ea'],st['esq'];b1,b2=g['betas'];st['step']+=1;step=st['step']
                    ea.mul_(b1).add_(gr,alpha=1-b1);esq.mul_(b2).addcmul(gr,gr,value=1-b2)
                    sk=(ea/(1-b1**step)).norm().item();fk=(esq/(1-b2**step)).sqrt().mean().item()
                    s._fs+=fgr*(fk-s._fs)
                    if s._fs>0: s._d=max(s._d,sk/(s._fs*(step**0.25)))
                    d=s._d;lr=g['lr']/d if d>0 else g['lr'];lr=min(lr,1.)
                    den=(esq/(1-b2**step)).sqrt().add(1e-8);upd=(ea/(1-b1**step))/den
                    if g['weight_decay']: p.data.add_(p.data,alpha=-lr*g['weight_decay'])
                    p.data.add_(upd,alpha=-lr)
            return closure() if closure else None
    
    class Adan(torch.optim.Optimizer):
        def __init__(s,params,lr=1e-3,betas=(0.98,0.92,0.99),eps=1e-8,wd=0):
            super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=wd))
        @torch.no_grad()
        def step(s,closure=None):
            for g in s.param_groups:
                lr,(b1,b2,b3),eps,wd=g['lr'],g['betas'],g['eps'],g['weight_decay']
                for p in g['params']:
                    if p.grad is None: continue
                    gr=p.grad.data;st=s.state[p]
                    if len(st)==0: st['m']=torch.zeros_like(p);st['v']=torch.zeros_like(p);st['n']=torch.zeros_like(p);st['step']=0
                    m,v,n=st['m'],st['v'],st['n'];st['step']+=1;t=st['step']
                    m.mul_(b1).add_(gr,alpha=1-b1);v.mul_(b2).addcmul(gr,gr,value=1-b2)
                    gp=gr+b3*(p.data-st.get('pp',p.data));st['pp']=p.data.clone()
                    n.mul_(b3).addcmul(gp,gp,value=1-b3)
                    mh=m/(1-b1**t);vh=v/(1-b2**t);nh=n/(1-b3**t)
                    if wd: p.data.add_(p.data,alpha=-lr*wd)
                    p.data.addcdiv_(mh,vh.sqrt().add(eps),value=-lr);p.data.addcdiv_(gp,nh.sqrt().add(eps),value=-lr)
            return closure() if closure else None
    
    class RAdam(torch.optim.Optimizer):
        def __init__(s,params,lr=1e-3,betas=(0.9,0.999),eps=1e-8,wd=0):
            super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=wd))
        @torch.no_grad()
        def step(s,closure=None):
            for g in s.param_groups:
                lr,(b1,b2),eps,wd=g['lr'],g['betas'],g['eps'],g['weight_decay']
                for p in g['params']:
                    if p.grad is None: continue
                    gr=p.grad.data;st=s.state[p]
                    if len(st)==0: st['step']=0;st['ea']=torch.zeros_like(p);st['esq']=torch.zeros_like(p)
                    ea,esq=st['ea'],st['esq'];st['step']+=1;step=st['step']
                    ea.mul_(b1).add_(gr,alpha=1-b1);esq.mul_(b2).addcmul(gr,gr,value=1-b2)
                    bc1,bc2=1-b1**step,1-b2**step;ri=2/(1-b2)-1;rt=ri-2*step*b2**step/bc2
                    if rt>4:
                        rv=math.sqrt((rt-4)*(rt-2)/ri/(ri-4));vh=(esq/bc2).sqrt().add_(eps)
                        p.data.addcdiv_(ea/bc1,vh,value=-lr*rv)
                    else: p.data.add_(ea/bc1,alpha=-lr)
                    if wd: p.data.add_(p.data,alpha=-lr*wd)
            return closure() if closure else None
    
    def get_opt(name,params,lr,wd):
        n=name.lower().replace('-','').replace('_','')
        if n=='adamw': return torch.optim.AdamW(params,lr=lr,weight_decay=wd,betas=(0.9,0.95))
        elif n=='lion': return Lion(params,lr=lr,wd=wd)
        elif n=='sophia': return Sophia(params,lr=lr,wd=wd)
        elif n=='adafactor': return torch.optim.AdamW(params,lr=lr,weight_decay=wd)  # Simplified
        elif n=='adan': return Adan(params,lr=lr,wd=wd)
        elif n=='radam': return RAdam(params,lr=lr,wd=wd)
        elif n=='muon': return Muon(params,lr=lr,wd=wd)
        elif n=='prodigy': return Prodigy(params,lr=lr,wd=wd)
        elif n.startswith('sf'): return SFAdamW(params,lr=lr,wd=wd)
        elif n.startswith('d'): return DAdam(params,lr=lr,wd=wd)
        else: return torch.optim.AdamW(params,lr=lr,weight_decay=wd)
    
    # ===== MODEL =====
    class Model(nn.Module):
        def __init__(s):
            super().__init__()
            s.embed=nn.Embedding(cfg.vocab_size,cfg.hidden_dim)
            s.pos=nn.Embedding(cfg.max_seq_len,cfg.hidden_dim)
            el=nn.TransformerEncoderLayer(d_model=cfg.hidden_dim,nhead=cfg.num_heads,
                dim_feedforward=int(cfg.hidden_dim*cfg.mlp_ratio),dropout=0.001,
                activation='gelu',batch_first=True,norm_first=True)
            s.enc=nn.TransformerEncoder(el,num_layers=cfg.num_layers)
            s.ln=nn.LayerNorm(cfg.hidden_dim);s.head=nn.Linear(cfg.hidden_dim,cfg.vocab_size,bias=False)
            s.head.weight=s.embed.weight
            s.apply(s.init)
            logger.info(f"Model: {sum(p.numel() for p in s.parameters())/1e6:.1f}M params")
        def init(s,m):
            if isinstance(m,nn.Linear): nn.init.normal_(m.weight,0,0.02)
            if hasattr(m,'bias') and m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Embedding): nn.init.normal_(m.weight,0,0.02)
        def forward(s,x,labels=None):
            B,T=x.shape;pos=torch.arange(T,device=x.device).unsqueeze(0)
            out=s.embed(x)+s.pos(pos);mask=torch.triu(torch.ones(T,T,device=x.device)*float('-inf'),diagonal=1)
            out=s.enc(out,mask=mask);out=s.ln(out);logits=s.head(out)
            r={"logits":logits}
            if labels is not None:
                sl=logits[...,:-1,:].contiguous();lb=labels[...,1:].contiguous()
                r["loss"]=F.cross_entropy(sl.view(-1,cfg.vocab_size),lb.view(-1))
            return r
    
    def train_one(opt_key,opt_name,opt_desc,lr,wd):
        dev="cuda" if torch.cuda.is_available() else "cpu"
        m=Model().to(dev);opt=get_opt(opt_key,m.parameters(),lr,wd)
        ts=cfg.num_epochs*15
        
        def lr_schedule(step):
            if step<cfg.warmup_steps: return step/max(1,cfg.warmup_steps)
            pr=(step-cfg.warmup_steps)/max(1,ts-cfg.warmup_steps);return max(0.1,0.5*(1+math.cos(math.pi*pr)))
        
        sch=torch.optim.lr_scheduler.LambdaLR(opt,lr_schedule)
        dl=[{"input_ids":torch.randint(0,cfg.vocab_size,(cfg.batch_size,cfg.max_seq_len),device=dev),
             "labels":torch.randint(0,cfg.vocab_size,(cfg.batch_size,cfg.max_seq_len),device=dev)} for _ in range(15)]
        hist=[];t0=time.time();bl=float('inf');scaler=torch.amp.GradScaler('cuda')
        for ep in range(cfg.num_epochs):
            el=0
            for b in dl:
                opt.zero_grad()
                with torch.amp.autocast('cuda'):
                    o=m(b['input_ids'],labels=b['labels']);l=o["loss"]
                if torch.isnan(l): return {"error":"NaN","opt":opt_key}
                scaler.scale(l).backward();scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.max_grad_norm)
                scaler.step(opt);scaler.update();sch.step();el+=l.item()
            hist.append(el/len(dl));bl=min(bl,hist[-1])
        del m;torch.cuda.empty_cache()
        return {"opt":opt_key,"name":opt_name,"desc":opt_desc,"lr":lr,"wd":wd,
                "fl":hist[-1],"bl":hist,"t":time.time()-t0,"conv":hist[-1]<hist[0]*0.8}
    
    # ===== MAIN LOOP =====
    all_res={};t0=time.time()
    for key,name,desc in ALL_OPTIMIZERS:
        logger.info(f"\n--- Testing {name} ({desc}) ---")
        res=[];study=optuna.create_study(direction='minimize',sampler=TPESampler(seed=42),
                                         pruner=MedianPruner(n_startup_trials=1,n_warmup_steps=5))
        def obj(trial):
            lr=trial.suggest_float('lr',*cfg.lr_range,log=True);wd=trial.suggest_float('wd',*cfg.wd_range)
            r=train_one(key,name,desc,lr,wd)
            if 'error' in r: return float('inf')
            res.append(r);return r['fl']
        try: study.optimize(obj,n_trials=cfg.n_trials,timeout=120)
        except Exception as e: logger.error(f"{name} error: {e}")
        
        ct=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE and t.value!=float('inf')]
        if ct:
            bt=max(ct,key=lambda x:-x.value) if ct[0].value!=float('inf') else ct[0]
            all_res[key]={"name":name,"desc":desc,"best":{"loss":bt.value,"lr":bt.params.get('lr','N/A'),
                          "wd":bt.params.get('wd','N/A')},"n":len(ct),"res":res}
            logger.info(f"{name}: Loss={bt.value:.4f}, LR={bt.params.get('lr','N/A'):.2e}")
        else:
            all_res[key]={"name":name,"desc":desc,"best":{"loss":float('inf'),"lr":"N/A","wd":"N/A"},"n":0,"res":[]}
            logger.warning(f"{name}: All failed")
    
    # Results
    valid={k:v for k,v in all_res.items() if v['best']['loss']<float('inf')}
    ranked=sorted(valid.items(),key=lambda x:x[1]['best']['loss'])
    
    logger.info("\n"+"="*70);logger.info("FINAL RESULTS");logger.info("="*70)
    medals=["1st","2nd","3rd","4th","5th","6th","7th","8th","9th","10th"]
    for i,(k,v) in enumerate(ranked):
        logger.info(f"{medals[i]:<6}{v['name']:<14} Loss:{v['best']['loss']:<8.4f} LR:{v['best']['lr']:.2e} WD:{v['best']['wd']:.4f}")
    
    tt=time.time()-t0;logger.info(f"\nTotal time: {tt/60:.1f}min")
    
    output={"ver":"v3.1-fast","ts":time.strftime("%Y-%m-%d %H:%M"),"hw":{"dev":device,
            "gpu":torch.cuda.get_device_name(0) if device=="cuda" else "N/A"},
            "cfg":{"layers":cfg.num_layers,"dim":cfg.hidden_dim,"heads":cfg.num_heads,
                  "vocab":cfg.vocab_size,"seq":cfg.max_seq_len,"trials":cfg.n_trials},
            "time_s":tt,"results":all_res,"rank":[{"k":k,**v} for k,v in ranked]}
    return output


@app.local_entrypoint()
def main():
    print("Running Fast Optimizer Benchmark v3.1...")
    res=run_fast_benchmark.remote()
    p=f"/home/z/my-project/download/fast_benchmark_v31_{int(time.time())}.json"
    os.makedirs(os.path.dirname(p),exist_ok=True)
    with open(p,'w') as f: json.dump(res,f,indent=2,default=str)
    print(f"Saved to: {p}")
    if 'rank' in res and res['rank']:
        print("\nRankings:")
        for i,e in enumerate(res['rank']):
            m=["GOLD","SILV","BRONZE"][i] if i<3 else f"#{i+1}"
            print(f"  {m}: {e['name']} (Loss={e['best']['loss']:.4f})")

if __name__=="__main__":
    main()
