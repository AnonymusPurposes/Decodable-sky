"""Layer-group validation, and the residual hook's live replacement.

The hook test needs PyTorch and is skipped where it is unavailable.
"""
import importlib.util
import unittest
import numpy as np
from core import Chart,group_label,layer_groups,trace_records,harmonics


class GroupTests(unittest.TestCase):
    def test_groups_and_final_normalization_guard(self):
        self.assertEqual(layer_groups(dict(layers=[60,66])),[[60],[66]])
        ls=list(range(46,88))
        self.assertEqual(layer_groups(dict(layers=ls,layer_groups=[ls])),[ls])
        self.assertEqual(group_label(ls),'46-87')
        self.assertEqual(group_label([60]),60)
        with self.assertRaises(ValueError):layer_groups(dict(layers=[87,88],layer_groups=[[87,88]]))


@unittest.skipUnless(importlib.util.find_spec('torch'),'PyTorch available on the GPU host')
class HookTests(unittest.TestCase):
    def test_joint_replaces_live_state_once_and_only_last_token(self):
        import torch
        from types import SimpleNamespace
        from run import Hook
        class Affine(torch.nn.Module):
            def forward(self,x):return 2*x+1
        layers=torch.nn.ModuleList([Affine(),Affine()])
        model=SimpleNamespace(model=SimpleNamespace(layers=layers))
        c=Chart(mu=np.zeros(16),sigma=np.ones(16),pca_mean=np.zeros(16),V=np.eye(16),b=np.zeros(16),W=np.eye(16)[:,:8])
        hooks=Hook(model,[1,2]);hooks.enabled=True;hooks.charts={1:c,2:c};hooks.targets=[[1,0,0],[0,1,0]]
        x=torch.arange(96,dtype=torch.float32).reshape(2,3,16)/20
        first=layers[0](x); second=layers[1](first)
        for i,target in enumerate(hooks.targets):
            desired=harmonics(target)
            np.testing.assert_allclose(c.scores(second[i,-1].numpy())[:8],desired,atol=1e-6)
            np.testing.assert_allclose(hooks.capture[2][i],(2*first[i,-1]+1).numpy(),atol=1e-6)
        torch.testing.assert_close(second[:,:-1],(4*x+3)[:,:-1])
        self.assertTrue(all([t['layer'] for t in ts]==[1,2] for ts in hooks.records()))
        # A later generation step must bypass both patches even though hooks
        # remain installed. A repeated cached source delta would fail this.
        next_x=torch.ones(2,1,16)
        torch.testing.assert_close(layers[1](layers[0](next_x)),4*next_x+3)
        self.assertTrue(all(len(ts)==2 for ts in hooks.records()))
        hooks.clear();hooks.enabled=True;hooks.charts={1:c,2:c}
        torch.testing.assert_close(layers[1](layers[0](x)),4*x+3)
        self.assertTrue(all(len(ts)==2 for ts in hooks.records()))
        for handle in hooks.handles:handle.remove()


if __name__=='__main__':unittest.main()
