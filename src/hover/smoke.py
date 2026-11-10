"""Check backbone and adapter integration on prepared windows."""
import time
from pathlib import Path
import numpy as np
import torch
from .backbones import build
from .experiments import Experiment
from .io import save, sha256
from .method import HOVER, Simple, adapter_loss
from .training import atomic_torch, state_digest

def smoke(dataset,data_root,output,backbone='NLINEAR_PAIRED',horizon=96,device='cpu',external_root=None):
    if backbone not in ('NLINEAR_PAIRED','OLINEAR'):
        raise ValueError('Technical verification scope is NLinear/OLinear')
    output=Path(output)
    if (output/'technical_smoke.json').exists():
        raise FileExistsError('Smoke evidence already exists; audit/reuse it rather than overwrite')
    started=time.monotonic()
    engine=Experiment(data_root,output,device,external_root)
    data=engine.data(dataset,horizon,'inner')
    q=engine.q(data) if backbone=='OLINEAR' else None
    args,_=engine.native_recipe(backbone,dataset,731,'inner')
    cuda=str(device).startswith('cuda')
    if cuda:torch.cuda.reset_peak_memory_stats()
    base=build(backbone,dataset,731,horizon,device,external_root,q,data,args)
    before=state_digest(base)
    if len(data.origins['TRAIN'])<64:raise ValueError('Full batch64 unavailable')
    x,y,xm,ym=data.batch('TRAIN',np.arange(64))
    rows=[]
    for arm,constructor in [('HOVER',HOVER),('SIMPLE',Simple)]:
        model=constructor(base,731,horizon=horizon).to(device)
        model.train();opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        prediction,frozen,_=model.components(x,xm,ym)
        if not torch.equal(prediction,frozen):raise ValueError('Zero initialization changed')
        loss,_,_=adapter_loss(prediction,y,frozen,1.)
        if not torch.isfinite(loss):raise ValueError('Nonfinite smoke loss')
        opt.zero_grad();loss.backward()
        if any(p.grad is not None for p in base.parameters()):raise ValueError('Frozen base received gradients')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        opt.step();model.eval()
        with torch.no_grad():expected=model(x,xm,ym)
        if not torch.isfinite(expected).all():raise ValueError('Nonfinite smoke prediction')
        path=output/(arm+'_technical_checkpoint.pt')
        path.parent.mkdir(parents=True,exist_ok=True)
        atomic_torch(path,dict(state=model.state_dict(),optimizer=opt.state_dict(),data=data.manifest(),Q=None if q is None else q['SHA256']))
        recovered=constructor(base,731,horizon=horizon).to(device)
        saved=torch.load(path,map_location='cpu');recovered.load_state_dict(saved['state'],strict=True)
        recovered.eval()
        with torch.no_grad():actual=recovered(x,xm,ym)
        if not torch.equal(expected,actual) or state_digest(base)!=before:raise ValueError('Checkpoint/frozen recovery mismatch')
        rows.append(dict(arm=arm,finite_loss=float(loss),optimizer_updates=1,checkpoint_SHA256=sha256(path),
            batch=64,valid_elements=int(y.numel()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            output_shape=list(actual.shape),frozen_unchanged=True,recovery_max_abs=0.))
    if cuda:torch.cuda.synchronize()
    result=dict(status='PASS',kind='TECHNICAL_SMOKE_NOT_FORMAL_FIT',dataset=dataset,backbone=backbone,horizon=horizon,
        data=data.manifest(),Q=None if q is None else q['SHA256'],rows=rows,wall_seconds=time.monotonic()-started,
        peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else 0,TEST_access=False)
    save(output/'technical_smoke.json',result)
    return result
