FROM boinc/client:latest

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu128
ARG TORCH_EXTRA_INDEX_URL=https://pypi.org/simple

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates python3-numpy python3-requests \
    && python3 -m pip install --no-cache-dir --break-system-packages --pre torch torchvision --index-url ${TORCH_EXTRA_INDEX_URL} --extra-index-url ${TORCH_INDEX_URL} --timeout 120 --retries 10 \
    && rm -rf /var/lib/apt/lists/*
