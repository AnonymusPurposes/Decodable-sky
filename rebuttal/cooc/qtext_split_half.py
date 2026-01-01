#!/usr/bin/env python
"""Split-half reliability of the corpus RDMs (Issue 3). Run from the repo root; reads the merged sets shipped in rebuttal/data/cooc."""
import sys,csv
sys.path.insert(0,'rebuttal/cooc')
import numpy as np
import qtext_spectrum as Q
cat=list(csv.DictReader(open('data/astro_objects.csv',encoding='utf-8')))
names=[r['name'] for r in cat]; types=np.array([r['type'] for r in cat])
fobj={}
for r in csv.DictReader(open('rebuttal/cooc/surface_forms_v2.csv',encoding='utf-8')): fobj.setdefault(r['form'],r['object'])
nodes=np.where(types=='constellation')[0]; ii,jj=np.triu_indices(len(nodes),1)
rows=[]
def R(f,var,kind,factor,k):
    c,u,n,_,_=Q.counts(f,var,names,fobj); kk=np.where(u>=Q.MIN_U)[0]
    m=Q.stat(c,u,n,kind,'zero')[np.ix_(kk,kk)]; w,v=np.linalg.eigh(.5*(m+m.T))
    return Q.rdm(w,v,kk,factor,k,len(names))[np.ix_(nodes,nodes)][ii,jj]
for a,b,l in (('rebuttal/data/cooc/sets/wiki_h0.npz','rebuttal/data/cooc/sets/wiki_h1.npz','wiki'),('rebuttal/data/cooc/sets/web_h0.npz','rebuttal/data/cooc/sets/web_h1.npz','web 187B'),('rebuttal/data/cooc/sets_ext/ext_h0.npz','rebuttal/data/cooc/sets_ext/ext_h1.npz','ext 975B')):
    for var in ('art','raw'):
        for kind in ('mstar','mstar_tt'):
            for k in (None,10):
                x,y=R(a,var,kind,'positive',k),R(b,var,kind,'positive',k); ok=~(np.isnan(x)|np.isnan(y))
                r=round(Q.corr(x[ok],y[ok]),3); print(l,var,kind,k,r); rows.append(dict(corpus=l,count='gated' if var=='art' else 'pure',statistic=kind,k=k or 'all',split_half_r=r))
with open('rebuttal/results/qtext_split_half.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
