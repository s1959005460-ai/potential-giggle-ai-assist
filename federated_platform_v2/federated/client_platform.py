import grpc, logging, time, numpy as np
from federated import federated_pb2 as pb, federated_pb2_grpc as pb_grpc
from federated.grpc_utils import serialize_state_dict_to_modelupdate, numpy_from_tensorproto
from trainer import get_trainer
import torch
from tenacity import retry, stop_after_attempt, wait_exponential

class GRPCRetryHelper:
    @staticmethod
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def call_rpc(fn, *args, **kwargs):
        return fn(*args, **kwargs)

class FederatedClientPlatform:
    def __init__(self, client_id, server_addr, trainer_name, trainer_kwargs, cfg):
        self.client_id = client_id
        self.server_addr = server_addr
        self.cfg = cfg
        self.grpc_opts = [('grpc.max_send_message_length', 50 * 1024 * 1024),
                          ('grpc.max_receive_message_length', 50 * 1024 * 1024)]
        self.channel = grpc.insecure_channel(server_addr, options=self.grpc_opts)
        self.stub = pb_grpc.FederatedServiceStub(self.channel)
        self.trainer = get_trainer(trainer_name, **trainer_kwargs)

    def pull_global_model(self):
        resp = GRPCRetryHelper.call_rpc(self.stub.GetGlobalModel, pb.Empty(), timeout=10)
        state = {}
        for t in resp.tensors:
            state[t.name] = numpy_from_tensorproto(t)
        server_round = int(resp.round)
        return state, server_round

    def push_update(self, round_idx, sample_count=0, meta=None):
        state = {k: v.detach().cpu().numpy() for k, v in self.trainer.model.state_dict().items()}
        mu = serialize_state_dict_to_modelupdate(state, self.client_id, round=round_idx, sample_count=sample_count, meta=meta or {})
        GRPCRetryHelper.call_rpc(self.stub.SendModelUpdate, mu, timeout=10)

    def run_one_round(self):
        state, server_round = self.pull_global_model()
        if state:
            # load into model if keys match
            try:
                self.trainer.model.load_state_dict({k: torch.from_numpy(v) for k,v in state.items()}, strict=False)
            except Exception as e:
                logging.warning(f"Failed to load global state: {e}")
        epochs = self.cfg.get('trainer',{}).get('local_epochs',1)
        for _ in range(epochs):
            self.trainer.train_one_epoch()
        sample_count = len(self.trainer.train_loader.dataset) if hasattr(self.trainer.train_loader, 'dataset') else 0
        self.push_update(server_round, sample_count=sample_count, meta=self.cfg.get('meta',{}))
