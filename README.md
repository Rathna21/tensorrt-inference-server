# TensorRT Inference Server using gRPC client server architecture #

The project is about building an inference server for deep learning model inference using NVIDIA Triton Inference sever(formerly known as TensorRT Inference Server). NVIDIA Triton Inference Server provides cloud inferencing solution optimized for NVIDIA GPUs.The server provides an inference service via an HTTP or GRPC endpoint, allowing remote clients to request inferencing for any model being managed by the server. For edge deployments, Triton Server is also available as a shared library with an API that allows the full functionality of the server to be included directly in an application.[official docs](https://docs.nvidia.com/deeplearning/sdk/triton-inference-server-guide/docs/index.html)

## Features of NVIDIA Triton Inference Server: ##

- Multiple framework support.
- Concurrent model execution support.
- Batching support. 
- Custom backend support. 
- Ensemble support. 
- Multi-GPU support. 
- Model repositories may reside on a locally accessible file system (e.g. NFS), in Google Cloud Storage or in Amazon S3.
- Readiness and liveness health endpoints suitable for any orchestration or deployment framework, such as Kubernetes.
- Metrics indicating GPU utilization, server throughput, and server latency.
- C library inferface allows the full functionality of Triton Server to be included directly in an application.

## How to run the project? ##

- git clone https://github.com/Rathna21/tensorrt-inference-server.git
- Install grpcio-tools and run the command to generate client and server stubs using protocol buffer compiler:
  - pip install grpcio-tools
  - python -m grpc_tools.protoc -I./protos --python_out=./client_server/inf_ser/stubs/ --grpc_python_out=./client_server/inf_ser/stubs/ ./protos/inference.proto
- Download the Tensorrt Inference Server docker image and run:
  - docker pull nvcr.io/nvidia/tritonserver:19.04-py3
- Download v1.4.0_ubuntu1804.clients.tar file from [repo] (https://github.com/NVIDIA/triton-inference-server/releases) and unzip the file.
- cp ./v1.4.0_ubuntu1804.clients.tar/python/tensorrtserver-1.4.0-py2.py3-none-linux_x86_64.whl ./client_server/whl/
- Add vgg16 onnx model from the model-zoo to the directory : ./models/vgg16/1/
- cd ./client_server
- docker-compose build
- docker-compose up
- python inference-client.py --img_path ./images/car.jpeg



