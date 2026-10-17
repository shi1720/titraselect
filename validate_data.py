"""Independent alignment check against the source authors' stored curve fits."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from titraselect import load_task,TASKS
p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('results/alignment.json'));args=p.parse_args()
checks=[]
for task in TASKS:
 raw,fits,grids,cut=load_task(args.data,task)
 for rep,(r,f,c) in enumerate(zip(raw,fits,grids)):
  f=f.loc[(f[4]>.8)&(f[5]<1)].head(1000)
  y=r.loc[f.index,list(range(1,len(c)+1))].to_numpy()
  cp=c.copy();cp[-1]=1e-20 if task.startswith('CH65') else 1e-14
  pred=np.log10(f[2].to_numpy()[:,None]*cp/(10**(-f[1].to_numpy())[:,None]+cp)+f[3].to_numpy()[:,None])
  computed=1-((pred-y)**2).sum(1)/((y-y.mean(1)[:,None])**2).sum(1)
  difference=float(np.nanmax(abs(computed-f[4].to_numpy())))
  assert difference<1e-7,(task,rep,difference)
  checks.append(dict(task=task,source=rep,curves=len(f),max_absolute_r2_difference=difference))
args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(checks,indent=2))
print('Aligned',sum(x['curves'] for x in checks),'curves across',len(checks),'source tables; max difference',max(x['max_absolute_r2_difference'] for x in checks))
