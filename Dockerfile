ARG MILES_BASE_IMAGE=radixark/miles:latest-cu12@sha256:efc8027fc47aaa9687dc4f1046093ed4e2f9789e52a932fcefb7031402aeff37
FROM ${MILES_BASE_IMAGE}

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
