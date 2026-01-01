import unittest
import numpy as np
from core import Chart, fit_forward, harmonics, unit, display, parse_answer


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(42)

    def test_harmonic_orthogonality_and_parity(self):
        u=unit(self.rng.normal(size=(100000,3)))
        phi=harmonics(u)
        np.testing.assert_allclose(phi.T@phi/len(phi),np.eye(8),atol=.016)
        np.testing.assert_allclose(harmonics(-u)[:,:3],-phi[:,:3])
        np.testing.assert_allclose(harmonics(-u)[:,3:],phi[:,3:])

    def test_forward_recovers_known_curved_map(self):
        u=unit(self.rng.normal(size=(60,3)))
        b=self.rng.normal(size=16); W=self.rng.normal(size=(16,8))
        bh,Wh=fit_forward(u,b+harmonics(u)@W.T)
        np.testing.assert_allclose(b,bh,atol=1e-12)
        np.testing.assert_allclose(W,Wh,atol=1e-12)

    def test_replacement_target_and_full_complement(self):
        for k in [16,128]:
            n=160
            V=np.linalg.qr(self.rng.normal(size=(n,k)))[0].T.astype(np.float32).astype(float)
            c=Chart(mu=self.rng.normal(size=n),sigma=np.exp(self.rng.normal(size=n)),pca_mean=self.rng.normal(size=n),
                    V=V,b=self.rng.normal(size=k),W=self.rng.normal(size=(k,8)))
            h=self.rng.normal(size=n); target=unit(self.rng.normal(size=3))
            after=c.replace(h,target); trace=c.trace(h,after,target)
            self.assertLess(trace['target_relative_error'],1e-12)
            self.assertLess(trace['pc_complement_change_norm'],1e-11)
            self.assertLess(trace['outside_retained_pcs_change_norm'],1e-11)
            np.testing.assert_allclose(c.replace(after,target),after,atol=1e-11)

    def test_self_replacement_is_not_generally_noop(self):
        k=16
        c=Chart(mu=np.zeros(k),sigma=np.ones(k),pca_mean=np.zeros(k),V=np.eye(k),b=self.rng.normal(size=k),W=self.rng.normal(size=(k,8)))
        u=unit([1,2,3]); on=c.b+harmonics(u)@c.W.T
        np.testing.assert_allclose(c.replace(on,u),on,atol=1e-12)
        off=on+c.Q[:,0]
        self.assertGreater(np.linalg.norm(c.replace(off,u)-off),.99)

    def test_constellation_answer_names(self):
        catalog=[dict(id=0,name='Crux constellation'),dict(id=1,name='Ursa Major constellation')]
        self.assertEqual(display(catalog[1]['name']),'Ursa Major')
        self.assertEqual(parse_answer('the Southern Cross.',catalog),0)
        self.assertEqual(parse_answer('Ursa Major.',catalog),1)
        self.assertIsNone(parse_answer('Ursa Major or Crux',catalog))


if __name__=='__main__':
    unittest.main()
