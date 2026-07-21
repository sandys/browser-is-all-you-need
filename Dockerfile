ARG MILES_BASE_IMAGE=radixark/miles:latest-cu12@sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37
FROM ${MILES_BASE_IMAGE}

# C++ reward workers use the benchmark's vendored Catch2 v2 harness. Boost is
# required by gigasecond/meetup; TBB is available for the parallel exercise.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      g++ \
      gawk \
      libboost-date-time-dev \
      libtbb-dev \
      util-linux \
    && rm -rf /var/lib/apt/lists/*

# Keep the Python package, precompiled cubins, and CUDA 12.9 JIT cache on the
# version required by the SGLang source bundled in the Miles base image.
ARG FLASHINFER_VERSION=0.6.12
ARG FLASHINFER_CUDA_INDEX=129
ENV FLASHINFER_VERSION=${FLASHINFER_VERSION}
RUN python3 -m pip install --no-cache-dir --no-deps --upgrade \
      "flashinfer-python==${FLASHINFER_VERSION}" \
      "flashinfer-cubin==${FLASHINFER_VERSION}" \
    && python3 -m pip install --no-cache-dir --no-deps --upgrade \
      "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
      --index-url "https://flashinfer.ai/whl/cu${FLASHINFER_CUDA_INDEX}" \
    && python3 -c 'from importlib.metadata import version; expected = "0.6.12"; packages = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"); actual = {name: version(name).split("+")[0] for name in packages}; assert all(item == expected for item in actual.values()), actual; print(actual)'

# The sglang-miles source in the base image moves ahead of the image build
# commit. Match its CUDA kernel guard and colocated offload dependency without
# replacing the base image's complete CUDA/PyTorch stack.
ARG SGLANG_KERNEL_VERSION=0.4.4
ARG TORCH_MEMORY_SAVER_VERSION=0.0.9.post1
ENV SGLANG_KERNEL_VERSION=${SGLANG_KERNEL_VERSION}
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      "sglang-kernel==${SGLANG_KERNEL_VERSION}" \
      --index-url "https://docs.sglang.ai/whl/cu${FLASHINFER_CUDA_INDEX}/" \
    && python3 -m pip install --no-cache-dir --no-deps --upgrade \
      "torch-memory-saver==${TORCH_MEMORY_SAVER_VERSION}" \
    && python3 -c 'from importlib.metadata import version; expected = {"sglang-kernel": "0.4.4", "torch-memory-saver": "0.0.9.post1"}; actual = {name: version(name).split("+")[0] for name in expected}; assert actual == expected, actual; print(actual)'

LABEL org.opencontainers.image.description="Miles GLM-4.7 H100 runtime with aligned FlashInfer packages"
