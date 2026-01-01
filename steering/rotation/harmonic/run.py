"""Single or joint-layer Mistral interventions, with optional sustained replacement."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from core import MODEL, REVISION, SEED, Chart, digest, display, harmonics,layer_groups,group_label


def event(out,status,**kw):
    row=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status=status,**kw)
    print(json.dumps(row),flush=True)
    with (out/'progress.jsonl').open('a') as f:
        f.write(json.dumps(row)+'\n')


class Hook:
    def __init__(self,model,layers):
        self.handles=[model.model.layers[l-1].register_forward_hook(self.make(l)) for l in layers]
        self.clear()

    def clear(self):
        self.seen=set(); self.charts={}; self.targets=None
        self.capture={}; self.traces=[]; self.enabled=False
        self.step=0; self.active=None; self.input_tokens=None; self.generation_calls=0

    def advance(self,active,input_tokens):
        """Begin a cached forward: patch the input token that predicts the next token."""
        self.step+=1; self.seen=set()
        self.active=list(active); self.input_tokens=list(input_tokens)

    def generation_step(self,module,args,kwargs,eos_ids):
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids is None:raise AssertionError('Generation must supply input_ids')
        latest=ids[:,-1].detach().cpu().tolist()
        if self.generation_calls:
            prior=self.active if self.active is not None else [True]*len(latest)
            self.advance([a and t not in eos_ids for a,t in zip(prior,latest)],latest)
        else:self.input_tokens=latest
        self.generation_calls+=1

    def position(self,i):
        return dict(step=self.step,phase='prefill' if self.step==0 else 'decode',
                    input_token_id=None if self.input_tokens is None else self.input_tokens[i])

    def records(self):
        return [ts[0] if len(ts)==1 else list(ts) for ts in self.traces]

    def make(self,layer):
        def apply(module,inputs,output):
            import torch
            if not self.enabled or layer in self.seen:
                return output
            self.seen.add(layer)
            tensor=output[0] if isinstance(output,tuple) else output
            live=tensor[:,-1,:].detach().float().cpu().numpy().astype(float)
            if self.step==0:self.capture[layer]=live
            if layer not in self.charts:
                return output
            chart=self.charts[layer]
            if not self.traces:self.traces=[[] for _ in live]
            if self.targets is None:  # untouched: return original tensor exactly
                for i,ts in enumerate(self.traces):
                    if self.active is None or self.active[i]:
                        ts.append(dict(layer=layer,delta_norm=0.,delta_relative_norm=0.,**self.position(i)))
                return output
            changed=tensor.clone()
            for i,target in enumerate(self.targets):
                if self.active is not None and not self.active[i]:continue
                ideal=chart.replace(live[i],target)
                ideal_trace=chart.trace(live[i],ideal,target)
                if ideal_trace['target_relative_error']>1e-9 or ideal_trace['pc_complement_change_norm']>1e-8:
                    raise AssertionError(f'Float64 projection failed: {ideal_trace}')
                corrected=torch.as_tensor(ideal,device=tensor.device,dtype=tensor.dtype)
                actual=corrected.float().cpu().numpy().astype(float)
                trace=chart.trace(live[i],actual,target)
                trace['layer']=layer
                trace.update(self.position(i))
                trace['ideal_target_relative_error']=ideal_trace['target_relative_error']
                trace['ideal_pc_complement_change_norm']=ideal_trace['pc_complement_change_norm']
                trace['ideal_outside_retained_pcs_change_norm']=ideal_trace['outside_retained_pcs_change_norm']
                if trace['target_relative_error']>.03:
                    raise AssertionError(f'Post-dtype-cast target miss exceeds 3%: {trace}')
                self.traces[i].append(trace)
                changed[i,-1,:]=corrected
            return (changed,*output[1:]) if isinstance(output,tuple) else changed
        return apply


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--max-new',type=int,default=24)
    ap.add_argument('--token-policy',choices=['prefill','sustained'],default='prefill')
    args=ap.parse_args()
    out=args.out; out.mkdir(parents=True,exist_ok=False)
    prep=args.prepared
    metadata=json.loads((prep/'manifest.json').read_text())
    groups=layer_groups(metadata)
    specs=json.loads((prep/'spec.json').read_text())
    catalog=json.loads((prep/'catalog.json').read_text())
    charts={(l,k):Chart(prep/f'chart_L{l}_K{k}.npz') for l in metadata['layers'] for k in metadata['ks']}
    snapshot=out/'inference_snapshot'; snapshot.mkdir()
    for p in Path(__file__).parent.iterdir():
        if p.suffix in ['.py','.sbatch','.md']:
            shutil.copy2(p,snapshot/p.name)
    os.environ.setdefault('HF_HOME','/data/work/hf_cache')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    torch.manual_seed(SEED)
    # Checkpoint and placement come from the prepared bundle; historical Mistral
    # bundles already carry model/revision, so their behaviour is unchanged.
    repo=metadata.get('model',MODEL); rev=metadata.get('revision',REVISION)
    if rev is None:
        # Bundles prepared before models_multi.py carried per-model revisions stored a
        # null here, which makes from_pretrained follow whatever main points at now.
        # Do not substitute REVISION: that is Mistral's commit and would be wrong for
        # every other repo. Re-run prepare_multi.py to pin the bundle.
        print(f'WARNING: bundle has no revision for {repo}; loading current main, which '
              f'may not be the checkpoint the published results used. Re-run '
              f'prepare_multi.py to pin it.',file=sys.stderr,flush=True)
    placement={'':0} if metadata.get('device_map','single')=='single' else 'auto'
    manifest=dict(model=repo,revision=rev,job=os.environ.get('SLURM_JOB_ID'),python=sys.version,
        packages={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','accelerate']},
        input_hashes={p.name:digest(p) for p in prep.iterdir() if p.is_file()},
        source_hashes={p.name:digest(p) for p in snapshot.iterdir()},max_new=args.max_new,
        scoring='Full candidate token log likelihood; cached continuation uses the same token policy as generation',
        token_policy=args.token_policy,
        batch_policy='Identical 16-prompt batches for every condition, no deduplication; separate matched natural-B batches',
        layer_groups=groups,
        intervention='Live b+(I-Pi)(z-b)+W phi(destination); each layer in selected group; '+
            ('last prompt token and subsequent active input tokens' if args.token_policy=='sustained' else 'last prompt token only'))
    event(out,'loading_model')
    tok=AutoTokenizer.from_pretrained(repo,revision=rev,local_files_only=True)
    tok.padding_side='left'
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(repo,revision=rev,dtype='auto',device_map=placement,
        attn_implementation='sdpa',local_files_only=True).eval()
    if metadata.get('keep_experts_impl'):
        # transformers 5.x swaps MoE grouped_mm -> batched_mm while decoding, which
        # copies the selected expert weights per token and exhausts memory at this batch.
        import contextlib
        model._optimize_model_for_decode=lambda:contextlib.nullcontext()
    in_dev=model.get_input_embeddings().weight.device
    manifest.update(dtype=str(model.dtype),device_map=str(getattr(model,'hf_device_map',{'':str(model.device)})),gpu=torch.cuda.get_device_name(0),placement=str(placement),
                    model_hidden_layers=model.config.num_hidden_layers,
                    eos_token_id=model.generation_config.eos_token_id,pad_token_id=tok.pad_token_id)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    hooks=Hook(model,metadata['layers'])
    preflight=tok([specs[0]['prompt']],return_tensors='pt').to(in_dev)
    hooks.enabled=True
    with torch.inference_mode():states=model(**preflight,output_hidden_states=True,use_cache=False).hidden_states
    boundary={l:float(np.max(np.abs(states[l][:,-1,:].float().cpu().numpy()-hooks.capture[l]))) for l in metadata['layers']}
    if max(boundary.values())>1e-6:raise AssertionError(f'Wrong hidden-state boundary: {boundary}')
    del states
    hooks.clear()
    checks=[]
    # Compatibility check uses the widest stored PCA width; identical to the
    # historical K=128 behaviour whenever 128 charts are present.
    kcheck=max(metadata['ks'])
    for check in json.loads((prep/'archive_checks.json').read_text()):
        enc=tok([check['prompt']],return_tensors='pt').to(in_dev)
        hooks.clear(); hooks.enabled=True
        with torch.inference_mode():model(**enc,use_cache=False)
        for l in metadata['layers']:
            z=charts[l,kcheck].scores(hooks.capture[l][0]); reference=np.array(check['scores'][str(l)])[:kcheck]
            rel=float(np.linalg.norm(z-reference)/np.linalg.norm(reference))
            checks.append(dict(prompt=check['prompt'],layer=l,relative_score_error=rel,
                               cosine=float(z@reference/(np.linalg.norm(z)*np.linalg.norm(reference)))))
            if rel>.08:raise AssertionError(f'Live/archive mismatch exceeds 8%: {checks[-1]}')
    hooks.clear()
    (out/'preflight.json').write_text(json.dumps(dict(hook_boundary_max_errors=boundary,archive_score_checks=checks),indent=2))
    event(out,'preflight_passed',archive_max_relative_error=max(c['relative_score_error'] for c in checks))

    def configure(group=None,k=None,arm=None):
        hooks.clear(); hooks.enabled=True
        if group is not None:
            hooks.charts={l:charts[l,k] for l in group}
            if arm!='untouched':
                hooks.targets=[catalog[r['source'] if arm=='self' else r['target']]['unit'] for r in specs]

    def generate(prompts,group=None,k=None,arm=None):
        enc=tok(prompts,return_tensors='pt',padding=True).to(in_dev)
        configure(group,k,arm)
        handle=None
        if args.token_policy=='sustained' and group is not None:
            eos=model.generation_config.eos_token_id
            eos_ids=set(eos if isinstance(eos,list) else [eos])
            handle=model.register_forward_pre_hook(lambda m,a,kw:hooks.generation_step(m,a,kw,eos_ids),with_kwargs=True)
        try:
            with torch.inference_mode():
                tokens=model.generate(**enc,do_sample=False,max_new_tokens=args.max_new,pad_token_id=tok.pad_token_id)
        finally:
            if handle is not None:handle.remove()
        generated=tokens[:,enc.input_ids.shape[1]:].cpu().tolist()
        result=[dict(completion=tok.decode(t,skip_special_tokens=True),tokens=t) for t in generated]
        traces=hooks.records(); captures={l:h.copy() for l,h in hooks.capture.items()}
        hooks.clear()
        return result,traces,captures

    def score(prompts,answers,group=None,k=None,arm=None):
        answer_ids=[]
        for p,a in zip(prompts,answers):
            prefix=tok.encode(p,add_special_tokens=True)
            full=tok.encode(p+a,add_special_tokens=True)
            if full[:len(prefix)]!=prefix or len(full)==len(prefix):
                raise AssertionError(f'Candidate retokenizes prompt boundary: {p!r}, {a!r}')
            answer_ids.append(full[len(prefix):])
        enc=tok(prompts,return_tensors='pt',padding=True).to(in_dev)
        configure(group,k,arm)
        hooks.input_tokens=enc.input_ids[:,-1].cpu().tolist()
        mask=enc.attention_mask
        values=[[] for _ in specs]
        with torch.inference_mode():
            output=model(**enc,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=True)
            sustained=args.token_policy=='sustained' and group is not None
            if not sustained:hooks.enabled=False
            for step in range(max(map(len,answer_ids))):
                current=torch.tensor([a[step] if step<len(a) else tok.pad_token_id for a in answer_ids],device=mask.device)
                logits=output.logits[:,-1,:].float()
                probs=logits.gather(1,current[:,None]).squeeze(1)-torch.logsumexp(logits,dim=-1)
                for i,v in enumerate(probs.cpu().tolist()):
                    if step<len(answer_ids[i]):values[i].append(v)
                if step+1==max(map(len,answer_ids)):break
                active=torch.tensor([[int(step<len(a))] for a in answer_ids],device=mask.device,dtype=mask.dtype)
                mask=torch.cat([mask,active],dim=1)
                if sustained:
                    hooks.advance([step+1<len(a) for a in answer_ids],current.cpu().tolist())
                output=model(input_ids=current[:,None],attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0)[:,-1:],
                    past_key_values=output.past_key_values,use_cache=True)
        traces=hooks.records()
        hooks.clear()
        return [dict(answer=a,token_ids=ids,token_logprobs=v,logprob=float(sum(v)),trace=traces[i] if traces else None)
                for i,(a,ids,v) in enumerate(zip(answers,answer_ids,values))]

    prompts=[r['prompt'] for r in specs]; target_prompts=[r['target_prompt'] for r in specs]
    answers=[[' '+display(catalog[r['candidates'][j]]['name']) for r in specs] for j in [0,1]]
    event(out,'baselines_start')
    baseline,_,captures=generate(prompts)
    natural,_,_=generate(target_prompts)
    base_scores=[score(prompts,a) for a in answers]
    natural_scores=[score(target_prompts,a) for a in answers]
    baseline_rows=[]
    for i,spec in enumerate(specs):
        row=dict(spec,baseline=baseline[i],natural_target=natural[i],
            baseline_scores=[s[i] for s in base_scores],natural_target_scores=[s[i] for s in natural_scores],
            live_fit_diagnostics={f'L{l}_K{k}':dict(
                source_distance=float(np.linalg.norm(charts[l,k].scores(captures[l][i])-charts[l,k].b-
                    harmonics(catalog[spec['source']]['unit'])@charts[l,k].W.T)))
                for l in metadata['layers'] for k in metadata['ks']})
        baseline_rows.append(row)
    (out/'baselines.json').write_text(json.dumps(baseline_rows,indent=2))
    completed=0
    with (out/'results.jsonl').open('x') as fout:
        for group in groups:
            label=group_label(group)
            for k in metadata['ks']:
                for arm in ['untouched','self','target']:
                    generated,traces,_=generate(prompts,group,k,arm)
                    scores=[score(prompts,a,group,k,arm) for a in answers]
                    if arm=='untouched':
                        if [r['tokens'] for r in generated]!=[r['tokens'] for r in baseline]:
                            raise AssertionError('Untouched hook changed generated tokens')
                        for j in [0,1]:
                            for i in range(len(specs)):
                                np.testing.assert_allclose(scores[j][i]['token_logprobs'],base_scores[j][i]['token_logprobs'],atol=1e-6,rtol=0)
                    for i,spec in enumerate(specs):
                        fout.write(json.dumps(dict(spec,layer=label,layers=group,k=k,arm=arm,**generated[i],trace=traces[i],scores=[s[i] for s in scores]))+'\n')
                    fout.flush(); completed+=len(specs)
                    event(out,'condition_complete',layer=label,k=k,arm=arm,done=completed,total=len(specs)*len(groups)*len(metadata['ks'])*3)
    event(out,'complete',trials=completed)


if __name__=='__main__':
    main()
