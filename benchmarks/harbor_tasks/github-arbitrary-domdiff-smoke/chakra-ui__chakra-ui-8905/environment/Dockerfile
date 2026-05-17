
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
    git remote add origin https://github.com/chakra-ui/chakra-ui.git && \
    (git fetch --depth 100 origin bc92c21762dded58134093e7d5bec5d366caced9 && git checkout bc92c21762dded58134093e7d5bec5d366caced9) || \
    (echo "Shallow clone failed, doing full clone..." && \
     git fetch origin && git checkout bc92c21762dded58134093e7d5bec5d366caced9)

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
RUN if ! git diff --quiet; then \
    git config user.email "benchmark@example.com" && \
    git config user.name "Benchmark Harness" && \
    git add -A && \
    HUSKY=0 git commit --no-verify -m "Apply benchmark harness"; \
fi

# Pre-build static assets at image build time
# No pre-build (dev-serve mode)

# Gold worktree for gold-signal capture (fix patch pre-applied + pre-built)
RUN git worktree add /testbed_gold HEAD \
    && cd /testbed_gold \
    && (git apply /testbed/patches/fix.patch || patch -p1 < /testbed/patches/fix.patch || true)
RUN ln -s /testbed/node_modules /testbed_gold/node_modules

# Supervisor entrypoint — keeps container alive, app restartable
COPY entrypoint.sh /entrypoint.sh
COPY restart_app.sh /restart_app.sh
RUN chmod +x /entrypoint.sh /restart_app.sh

# tini is already available in node base images; install if missing
RUN command -v tini >/dev/null 2>&1 || (apt-get update && apt-get install -y tini && rm -rf /var/lib/apt/lists/*)

EXPOSE 6006

ENTRYPOINT ["tini", "--"]
CMD ["/bin/sh", "-lc", "while true; do sleep 3600; done"]
