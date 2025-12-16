FROM boinc/client:latest

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 ca-certificates python3-numpy python3-requests \
    && rm -rf /var/lib/apt/lists/*
