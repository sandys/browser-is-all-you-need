
FROM node:20

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    patch \
    && rm -rf /var/lib/apt/lists/* \
    && corepack enable

# Install static file server
RUN npm install -g serve

WORKDIR /testbed

# Clone repository at base commit (buggy state)
RUN git init && \
    git remote add origin https://github.com/radix-ui/primitives.git && \
    (git fetch --depth 100 origin 7e2adc6dee70d175191ace6ce60e9aa86e46f2f2 && git checkout 7e2adc6dee70d175191ace6ce60e9aa86e46f2f2) || \
    (echo "Shallow clone failed, doing full clone..." && \
     git fetch origin && git checkout 7e2adc6dee70d175191ace6ce60e9aa86e46f2f2)

# Install dependencies
RUN corepack prepare pnpm@latest --activate && CYPRESS_INSTALL_BINARY=0 pnpm install --frozen-lockfile || CYPRESS_INSTALL_BINARY=0 pnpm install

# Copy evaluation script and patches
COPY evaluation.sh /testbed/evaluation.sh
RUN chmod +x /testbed/evaluation.sh
COPY patches/ /testbed/patches/
RUN if [ -f /testbed/patches/test.patch ]; then \
    git apply /testbed/patches/test.patch || \
    patch -p1 < /testbed/patches/test.patch || true; \
fi

# Pre-build static assets at image build time
# No pre-build (dev-serve mode)

# Gold worktree for gold-signal capture (fix patch pre-applied + pre-built)
RUN git worktree add /testbed_gold 7e2adc6dee70d175191ace6ce60e9aa86e46f2f2 \
    && cd /testbed_gold \
    && (git apply /testbed/patches/fix.patch || patch -p1 < /testbed/patches/fix.patch || true)
RUN ln -s /testbed/node_modules /testbed_gold/node_modules

# Supervisor entrypoint — keeps container alive, app restartable
COPY entrypoint.sh /entrypoint.sh
COPY restart_app.sh /restart_app.sh
RUN chmod +x /entrypoint.sh /restart_app.sh

# tini is already available in node base images; install if missing
RUN command -v tini >/dev/null 2>&1 || (apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*)

EXPOSE 9009

ENTRYPOINT ["tini", "--"]
CMD ["/bin/sh", "-lc", "while true; do sleep 3600; done"]
