import grpc
from concurrent import futures
import time, logging, os, pickle
from threading import Lock, Timer
import numpy as np
from federated import federated_pb2_grpc as pb_grpc
from federated import federated_pb2 as pb
from federated.grpc_utils import deserialize_modelupdate_to_state_dict
from collections import defaultdict
from federated.round_manager import RoundManager
from controller.epsilon_audit import EpsilonAudit
from prometheus_client import start_http_server, Summary, Gauge, Counter

AGG_TIME = Summary('federated_aggregation_seconds','Time spent in aggregation')
PENDING = Gauge('federated_pending_updates','Pending updates count')
ROUND_COUNTER = Gauge('federated_round','Current federated round')
CLIENT_EPS = Gauge('federated_client_epsilon','Cumulative epsilon per client', ['client_id'])

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

class FederatedService(pb_grpc.FederatedServiceServicer):
    def __init__(self, cfg, model_store=None, init_model_fn=None):
        self.cfg = cfg
        self.lock = Lock()
        self.round_manager = RoundManager(start_round=cfg.get('server',{}).get('current_round',0))
        self.global_model = {}
        self.pending_updates = []
        self.model_store = model_store or {}
        self.eps_reports = []
        self.init_model_fn = init_model_fn
        self.audit = EpsilonAudit(max_epsilon=cfg.get('privacy',{}).get('max_epsilon',10.0))
        self._load_state_or_init()
        # start prometheus exporter if configured
        metrics_port = cfg.get('metrics',{}).get('port', None)
        if metrics_port:
            start_http_server(metrics_port)
        self.aggregation_timer = None
        self._start_aggregation_timer_if_needed()

    def _persistence_dir(self):
        return self.cfg.get('server',{}).get('persistence_path','./federated_persist')

    def _global_model_file(self):
        return os.path.join(self._persistence_dir(), self.cfg.get('server',{}).get('global_model_file','global_model.pkl'))

    def _eps_file(self):
        return os.path.join(self._persistence_dir(), self.cfg.get('server',{}).get('eps_file','eps_reports.pkl'))

    def _ensure_persistence_dir(self):
        d = self._persistence_dir()
        os.makedirs(d, exist_ok=True)

    def _save_state(self):
        self._ensure_persistence_dir()
        with open(self._global_model_file(),'wb') as f:
            pickle.dump({'model':self.global_model,'round':self.round_manager.get_round()}, f)
        with open(self._eps_file(),'wb') as f:
            pickle.dump(self.eps_reports, f)
        logging.info('Persisted global model, round and eps reports to disk.')

    def _load_state_or_init(self):
        try:
            gmfile = self._global_model_file()
            epsfile = self._eps_file()
            if os.path.exists(gmfile):
                with open(gmfile,'rb') as f:
                    data = pickle.load(f)
                    self.global_model = data.get('model',{})
                    self.round_manager.set_round(data.get('round',0))
                logging.info('Loaded persisted global model and round.')
            else:
                init_spec = self.cfg.get('server',{}).get('initial_model', None)
                if isinstance(init_spec, str) and os.path.exists(init_spec):
                    with open(init_spec,'rb') as f:
                        self.global_model = pickle.load(f)
                    logging.info(f'Loaded initial model from {init_spec}')
                elif init_spec == 'from_scratch' and self.init_model_fn is not None:
                    self.global_model = self.init_model_fn()
                    logging.info('Initialized global model from init_model_fn()')
                else:
                    logging.info('No persisted model found and no init specified. Global model empty; waiting for first client update.')
            if os.path.exists(epsfile):
                with open(epsfile,'rb') as f:
                    self.eps_reports = pickle.load(f)
                for cid, eps, delta, rnd in self.eps_reports:
                    CLIENT_EPS.labels(client_id=cid).set(eps)
        except Exception as e:
            logging.warning(f'Failed to load persisted state: {e}')

    def _start_aggregation_timer_if_needed(self):
        timeout = self.cfg.get('server',{}).get('aggregation_timeout',None)
        if timeout:
            if self.aggregation_timer:
                self.aggregation_timer.cancel()
            self.aggregation_timer = Timer(timeout, self._on_aggregation_timeout)
            self.aggregation_timer.daemon = True
            self.aggregation_timer.start()

    def _on_aggregation_timeout(self):
        logging.info('Aggregation timeout fired.')
        with self.lock:
            if len(self.pending_updates) > 0:
                self.aggregate_and_update_global()
        self._start_aggregation_timer_if_needed()

    def SendModelUpdate(self, request, context):
        client_id = request.client_id
        round_idx = int(request.round)
        server_round = self.round_manager.get_round()
        if round_idx < server_round:
            logging.warning(f'Received stale update from {client_id}: client round {round_idx} < server round {server_round}. Ignoring.')
            return pb.Empty()
        if round_idx > server_round:
            logging.warning(f'Received future update from {client_id}: client round {round_idx} > server round {server_round}. Rejecting.')
            return pb.Empty()
        sample_count = int(request.sample_count) if request.sample_count else 0
        meta = dict(request.meta) if request.meta else {}
        state_dict = deserialize_modelupdate_to_state_dict(request)
        logging.info(f'Received update from {client_id} (round {round_idx}), sample_count={sample_count}, keys={list(state_dict.keys())[:5]}')
        with self.lock:
            self.pending_updates.append((client_id, round_idx, sample_count, state_dict, meta))
            PENDING.set(len(self.pending_updates))
            min_clients = int(self.cfg.get('server',{}).get('min_clients',3))
            if len(self.pending_updates) >= min_clients:
                logging.info(f'Min clients reached ({len(self.pending_updates)} >= {min_clients}). Aggregating now.')
                self.aggregate_and_update_global()
            else:
                self._start_aggregation_timer_if_needed()
        return pb.Empty()

    @AGG_TIME.time()
    def aggregate_and_update_global(self):
        if len(self.pending_updates) == 0:
            logging.info('No pending updates to aggregate.')
            return
        logging.info(f'Aggregating {len(self.pending_updates)} updates.')
        total_samples = sum(max(1,u[2]) for u in self.pending_updates)
        keys = set()
        for _,_,_,st,_ in self.pending_updates:
            keys.update(st.keys())
        accum = {k: None for k in keys}
        for client_id, rnd, sample_count, st, meta in self.pending_updates:
            w = max(1, sample_count)
            for k in keys:
                v = st.get(k, None)
                if v is None:
                    continue
                if accum[k] is None:
                    accum[k] = (v.astype(np.float64) * w)
                else:
                    accum[k] += (v.astype(np.float64) * w)
        new_global = {}
        for k, summed in accum.items():
            if summed is None:
                continue
            avg = (summed / float(total_samples)).astype(np.float32)
            new_global[k] = avg
        self.global_model = new_global
        self.pending_updates = []
        PENDING.set(0)
        # increment round after successful aggregation
        new_round = self.round_manager.increment()
        ROUND_COUNTER.set(new_round)
        try:
            self._save_state()
        except Exception as e:
            logging.warning(f'Failed to persist state: {e}')
        try:
            self.evaluate_global()
        except Exception as e:
            logging.warning(f'Evaluation hook failed: {e}')

    def evaluate_global(self):
        evaluator_name = self.cfg.get('evaluation',{}).get('evaluator', None)
        if evaluator_name is None:
            logging.debug('No evaluator provided in config; skipping evaluation.')
            return
        # dynamic evaluator lookup from registry
        from core.registry import get_evaluator
        evaluator = get_evaluator(evaluator_name)
        if evaluator is None:
            logging.warning(f'Evaluator {evaluator_name} not found in registry.')
            return
        logging.info('Evaluating global model...')
        metrics = evaluator(self.global_model)
        logging.info(f'Evaluation metrics: {metrics}')
        self.model_store.setdefault('eval_history', []).append(metrics)
        try:
            with open(os.path.join(self._persistence_dir(),'eval_history.pkl'),'wb') as f:
                pickle.dump(self.model_store.get('eval_history',[]), f)
        except Exception as e:
            logging.warning(f'Failed to persist eval_history: {e}')

    def GetGlobalModel(self, request, context):
        tensors = []
        for k, v in self.global_model.items():
            tensors.append(pb.TensorProto(data=np.asarray(v).tobytes(), shape=list(np.asarray(v).shape), dtype=str(np.asarray(v).dtype), name=k))
        return pb.GlobalModel(tensors=tensors, round=self.round_manager.get_round())

    def ReportEpsilon(self, request, context):
        cid, eps, delta, rnd = request.client_id, request.epsilon, request.delta, request.round
        logging.info(f'Epsilon report from {cid}: eps={eps:.4f}, delta={delta}, round={rnd}')
        # validate using conservative mode by default; advanced mode requires RDP accountant
        ok = self.audit.add_report(cid, eps, delta, rnd)
        CLIENT_EPS.labels(client_id=cid).set(self.audit.get_eps(cid))
        self.eps_reports.append((cid, eps, delta, rnd))
        try:
            with open(self._eps_file(),'wb') as f:
                pickle.dump(self.eps_reports, f)
        except Exception as e:
            logging.warning(f'Failed to persist eps_reports: {e}')
        return pb.Empty()

def serve(cfg, port=50051, init_model_fn=None):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = FederatedService(cfg, model_store={'epsilon_audit': EpsilonAudit()}, init_model_fn=init_model_fn)
    pb_grpc.add_FederatedServiceServicer_to_server(service, server)
    # TLS optional: cfg.security.tls keys if provided
    security = cfg.get('security',{})
    if security.get('tls',False):
        cert = security.get('cert_file'); key = security.get('key_file')
        with open(cert,'rb') as f: cert_chain = f.read()
        with open(key,'rb') as f: private_key = f.read()
        server_credentials = grpc.ssl_server_credentials(((private_key, cert_chain),))
        server.add_secure_port(f'[::]:{port}', server_credentials)
    else:
        server.add_insecure_port(f'[::]:{port}')
    server.start()
    logging.info(f'Federated server started on port {port}')
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
