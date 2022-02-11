# Deep Chimpact Competition Benchmark Code

This repo contains the code from the [benchmark blog post](https://www.drivendata.co/blog/deep-chimpact-benchmark/) for the [Deep Chimpact: Depth Estimation for Wildlife Conservation](https://www.drivendata.org/competitions/82/competition-wildlife-video-depth-estimation/page/390/) competition sponsored by MathWorks.

`BenchmarkBlog.mlx` is the MATLAB live script of the benchmark code with detailed instructions on how to load, process, and predict on the data. *Note: To access this file you have to download or clone the repo.* 

`BenchmarkCodeOnly.m` is the MATLAB code file of the benchmark, which is viewable directly on GitHub.

`pytorchToOnnx.py` ports a trained depth estimation model to ONNX so that it can be used in MATLAB.
