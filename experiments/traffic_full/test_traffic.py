import unittest, tempfile, pathlib, sys, numpy as np, torch
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from train import TrafficEnv,Client,SEEDS,METHODS,LATENT,OBS,aggregate
class TrafficContract(unittest.TestCase):
    def test_no_attack_flag(self):
        e=TrafficEnv(0,11,0,0)
        assert len(e.obs())==OBS==5
        e.t=15;assert len(e.obs())==5
    def test_exogenous_match(self):
        a,b=TrafficEnv(2,11,10,1),TrafficEnv(2,11,10,1)
        np.testing.assert_array_equal(a.arrivals,b.arrivals)
        np.testing.assert_array_equal(a.noise,b.noise)
    def test_attack_shift(self):
        e=TrafficEnv(1,11,0,0)
        assert e.arrivals.shape==(60,2)
    def test_two_phase_service_and_delay(self):
        e=TrafficEnv(0,11,0,0);e.t=15;e.prev_action=0
        q=e.q.copy()+e.arrivals[15];e.step(1)
        np.testing.assert_allclose(e.q,np.maximum(0,q-np.array([5.4,1.])))
    def test_prototype_and_local_encoding(self):
        c=Client(0,11,'AEFC-FRL',.001,8,.6)
        assert c.actor.encode(torch.zeros(2,5)).shape==(2,LATENT)
        no=Client(0,11,'AEFC-no-adapter',.001,8,.6)
        assert no.actor.encoder is None
    def test_model_methods(self):
        for method in METHODS:
            c=Client(0,11,method,.001,8,.6)
            x=c.episode(11,0,0,True,True)
            assert x['right_censored']==(x['recovery_step']==-1)
            assert np.isfinite(x['return'])
if __name__=='__main__':unittest.main()