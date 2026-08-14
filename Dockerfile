ARG MILES_BASE_IMAGE=radixark/miles:latest-cu12@sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37
FROM ${MILES_BASE_IMAGE}

# Keep the compiler and sandbox surface identical across Modal and SkyPilot.
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      bubblewrap \
      ca-certificates \
      cmake \
      curl \
      gawk \
      git \
      make \
      python3-venv \
      rsync \
      software-properties-common \
      util-linux \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      gcc-13 \
      g++-13 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100 \
    && rm -rf /var/lib/apt/lists/*

# Keep evaluation dependencies isolated from the Miles Python environment while
# making the same image usable for training and fixed-26 evaluation.
ARG GLM47_AIDER_COMMIT=5dc9490bb35f9729ef2c95d00a19ccd30c26339c
ARG GLM47_POLYGLOT_COMMIT=7e0611e77b54e2dea774cdc0aa00cf9f7ed6144f
RUN git clone https://github.com/Aider-AI/aider.git /aider \
    && git -C /aider checkout --detach "${GLM47_AIDER_COMMIT}" \
    && git clone https://github.com/Aider-AI/polyglot-benchmark.git \
      /aider/tmp.benchmarks/polyglot-benchmark \
    && git -C /aider/tmp.benchmarks/polyglot-benchmark checkout --detach \
      "${GLM47_POLYGLOT_COMMIT}" \
    && python3 -m venv /opt/aider-venv \
    && /opt/aider-venv/bin/pip install --no-cache-dir -e '/aider[dev]'

# Keep the Python package, precompiled cubins, and CUDA 12.9 JIT cache on the
# version required by the SGLang source bundled in the Miles base image.
ARG GLM47_FLASHINFER_VERSION=0.6.12
ARG GLM47_FLASHINFER_CUDA_INDEX=129
ENV FLASHINFER_VERSION=${GLM47_FLASHINFER_VERSION}
ENV FLASHINFER_CUDA_INDEX=${GLM47_FLASHINFER_CUDA_INDEX}
RUN python3 -m pip install --no-cache-dir --no-deps --upgrade --force-reinstall \
      "flashinfer-python==${FLASHINFER_VERSION}" \
      "flashinfer-cubin==${FLASHINFER_VERSION}" \
    && python3 -m pip install --no-cache-dir --no-deps --upgrade --force-reinstall \
      "flashinfer-jit-cache==${FLASHINFER_VERSION}" \
      --index-url "https://flashinfer.ai/whl/cu${FLASHINFER_CUDA_INDEX}" \
    && python3 -c 'from importlib.metadata import version; expected = "0.6.12"; packages = ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"); actual = {name: version(name).split("+")[0] for name in packages}; assert all(item == expected for item in actual.values()), actual; print(actual)'

# The sglang-miles source in the base image moves ahead of the image build
# commit. Match its CUDA kernel guard and colocated offload dependency without
# replacing the base image's complete CUDA/PyTorch stack.
ARG GLM47_SGLANG_KERNEL_VERSION=0.4.4
ARG GLM47_TORCH_MEMORY_SAVER_VERSION=0.0.9.post1
ENV SGLANG_KERNEL_VERSION=${GLM47_SGLANG_KERNEL_VERSION}
ENV TORCH_MEMORY_SAVER_VERSION=${GLM47_TORCH_MEMORY_SAVER_VERSION}
RUN python3 -m pip install --no-cache-dir --no-deps --force-reinstall \
      "sglang-kernel==${SGLANG_KERNEL_VERSION}" \
      --index-url "https://docs.sglang.ai/whl/cu${FLASHINFER_CUDA_INDEX}/" \
    && python3 -m pip install --no-cache-dir --no-deps --upgrade \
      "torch-memory-saver==${TORCH_MEMORY_SAVER_VERSION}" \
    && python3 -c 'from importlib.metadata import version; expected = {"sglang-kernel": "0.4.4", "torch-memory-saver": "0.0.9.post1"}; actual = {name: version(name).split("+")[0] for name in expected}; assert actual == expected, actual; print(actual)'

LABEL org.opencontainers.image.description="Miles GLM-4.7 H100 runtime with aligned FlashInfer packages"
