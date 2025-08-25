Federated Platform v2 - Enhanced package

This package includes:
- core/registry: unified registry for trainers, metrics, evaluators
- trainer/: BaseTrainer and example trainers (gnn, image)
- federated/: server_platform (RoundManager, init_model_fn support, FedAvg), client_platform (with retries)
- controller/epsilon_audit: integrated audit (conservative by default)
- metrics/defaults: metric registration
- hydra_config: example configuration (Hydra compatible)
- example.py: runnable in-process demo (no gRPC) showing aggregation & persistence
- basic tests and Dockerfile

Quick start:
1) Generate protobuf code: python -m grpc_tools.protoc -I./federated --python_out=./federated --grpc_python_out=./federated federated/federated.proto
2) Install deps: pip install -r requirements.txt
3) Run demo: python example.py
