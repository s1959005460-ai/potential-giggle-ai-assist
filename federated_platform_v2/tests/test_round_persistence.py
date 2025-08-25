import tempfile, os, pickle
from federated.server_platform import FederatedService
def test_persistence(tmp_path):
    cfg = {'server': {'persistence_path': str(tmp_path), 'global_model_file':'global_model.pkl', 'eps_file':'eps.pkl', 'current_round':0}}
    svc = FederatedService(cfg, model_store={})
    svc.global_model = {'w': __import__('numpy').array([1,2,3])}
    svc._save_state()
    # create new service to load state
    svc2 = FederatedService(cfg, model_store={})
    assert 'w' in svc2.global_model
    assert svc2.round_manager.get_round() == 0
