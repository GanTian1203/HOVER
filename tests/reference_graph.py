"""Reference implementations for numerical regression tests."""
import copy
import torch
from torch import nn
from torch.nn import functional as F

class Model(nn.Module):
 def __init__(self,base,seed,mode):
  super().__init__();self.mode=mode;self.base=copy.deepcopy(base)
  for z in self.base.parameters():z.requires_grad_(False)
  torch.manual_seed(seed)
  self.history=nn.Sequential(nn.Linear(336,64),nn.GELU(),nn.Linear(64,64),nn.GELU());self.head_h=nn.Linear(64,96)
  self.future=nn.Sequential(nn.Linear(96,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
  self.hist_proj=nn.Linear(64,32);self.future_proj=nn.Linear(32,32,bias=False);self.norm=nn.LayerNorm(32);self.head_q=nn.Linear(32,96)
  nn.init.zeros_(self.head_h.weight);nn.init.zeros_(self.head_h.bias);nn.init.zeros_(self.head_q.weight);nn.init.zeros_(self.head_q.bias)
 def parts(self,x):
  with torch.no_grad():b=self.base(x)
  z=x.transpose(1,2);mu=z.mean(-1,keepdim=True);sd=(z.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt();h=self.history((z-mu)/sd);dh=self.head_h(h)
  if self.mode=='HISTORY':dq=torch.zeros_like(dh);q=h[:,:,:32]*0
  else:
   coord=(b.transpose(1,2)-z[:,:,-1:])/sd
   if self.mode=='SURROGATE':coord=(z[:,:,-96:]-z[:,:,-1:])/sd
   a=self.future(coord);u=self.hist_proj(h);v=self.future_proj(a)
   q=self.norm(F.gelu(u+v) if self.mode=='CONCAT' else torch.tanh(u)*torch.tanh(v));dq=self.head_q(q)
  pred=b+((dh+dq)*sd).transpose(1,2)
  return pred,b,{'history_delta_rms':(dh*sd).square().mean().sqrt(),'conditional_delta_rms':(dq*sd).square().mean().sqrt(),'interaction_rms':q.square().mean().sqrt()}
 def forward(self,x):return self.parts(x)[0]
 def prune(self):
  if self.mode=='HISTORY':
   for key in ['future','hist_proj','future_proj','norm','head_q']:delattr(self,key)

Original=Model

class CalendarBridge(nn.Module):
    """Forward the known calendar features with the input window."""
    def __init__(self,backbone):super().__init__();self.network=backbone;self.xmark=None;self.future_mark=None
    def forward(self,x):return self.network(x,self.xmark,self.future_mark)


class ZeroHistoryDirect(nn.Module):
    def forward(self,h):return h.new_zeros((*h.shape[:-1],96))


class HOVER(Original):
    def __init__(self,backbone,seed,variant='full'):
        mapping={'full':'BILINEAR','history_only':'HISTORY','no_history_direct':'BILINEAR','concatenation':'CONCAT','history_condition_replacement':'SURROGATE'}
        assert variant in mapping
        super().__init__(CalendarBridge(backbone),seed,mapping[variant]);self.variant=variant
        if variant=='history_only':self.prune()
        if variant=='no_history_direct':self.head_h=ZeroHistoryDirect()
        self.base.eval()
    def train(self,mode=True):
        super().train(mode);self.base.eval();return self
    def components(self,x,xmark=None,future_mark=None):
        self.base.xmark=xmark;self.base.future_mark=future_mark
        return Original.parts(self,x)
    def forward(self,x,xmark=None,future_mark=None):return self.components(x,xmark,future_mark)[0]


class SIMPLE(nn.Module):
    """Reference shared MLP with a zero-initialized residual head."""
    def __init__(self,backbone,seed):
        super().__init__();self.base=copy.deepcopy(backbone)
        for p in self.base.parameters():p.requires_grad_(False)
        torch.manual_seed(seed);self.encoder=nn.Sequential(nn.Linear(432,70),nn.GELU(),nn.Linear(70,70),nn.GELU());self.head=nn.Linear(70,96)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias);self.base.eval()
    def train(self,mode=True):super().train(mode);self.base.eval();return self
    def components(self,x,xmark=None,future_mark=None):
        with torch.no_grad():b=self.base(x,xmark,future_mark)
        z=x.transpose(1,2);mu=z.mean(-1,keepdim=True);sd=(z.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt()
        coord=(b.transpose(1,2)-z[:,:,-1:])/sd
        delta=self.head(self.encoder(torch.cat([(z-mu)/sd,coord],-1)))*sd
        return b+delta.transpose(1,2),b,{}
    def forward(self,x,xmark=None,future_mark=None):return self.components(x,xmark,future_mark)[0]

