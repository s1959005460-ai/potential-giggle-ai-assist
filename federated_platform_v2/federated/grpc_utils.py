import numpy as np
from typing import Dict
from federated import federated_pb2 as pb

NUMPY_TO_PROTO = {
    'float32': 'float32',
    'float64': 'float64',
    'int32': 'int32',
    'int64': 'int64',
    'uint8': 'uint8'
}
PROTO_TO_NUMPY = {v:k for k,v in NUMPY_TO_PROTO.items()}

def tensorproto_from_numpy(name: str, arr: np.ndarray) -> pb.TensorProto:
    arr = np.asarray(arr)
    dtype_str = str(arr.dtype)
    if dtype_str not in NUMPY_TO_PROTO:
        # fallback to float32 with warning
        dtype_str = 'float32'
        arr = arr.astype(np.float32)
    return pb.TensorProto(data=arr.tobytes(), shape=list(arr.shape), dtype=dtype_str, name=name)

def numpy_from_tensorproto(tproto: pb.TensorProto) -> np.ndarray:
    dtype = PROTO_TO_NUMPY.get(tproto.dtype, 'float32')
    arr = np.frombuffer(tproto.data, dtype=dtype)
    if len(tproto.shape) > 0:
        arr = arr.reshape(tproto.shape)
    return arr

def serialize_state_dict_to_modelupdate(state_dict: Dict[str, np.ndarray], client_id: str, round: int=0, sample_count: int=0, meta: dict=None) -> pb.ModelUpdate:
    tensors = [tensorproto_from_numpy(name, arr) for name, arr in state_dict.items()]
    mu = pb.ModelUpdate(client_id=client_id, tensors=tensors, round=round, sample_count=sample_count)
    if meta:
        for k,v in meta.items():
            mu.meta[k] = v
    return mu

def deserialize_modelupdate_to_state_dict(model_update: pb.ModelUpdate) -> Dict[str, np.ndarray]:
    state = {}
    for t in model_update.tensors:
        state[t.name] = numpy_from_tensorproto(t)
    return state
