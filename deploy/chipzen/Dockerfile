# syntax=docker/dockerfile:1.7
#
# IP-protected Chipzen bot image (adapted from the Python starter).
#
# Multi-stage build that compiles bot.py AND the whole engine/ package into
# Cython .so modules in the builder stage, then ships only the compiled
# binaries in the runtime stage. The runtime image contains no readable .py
# source for the strategy code.
#
# Build:   docker build -t 0fold-heuristic:v1 .
# Export:  docker save 0fold-heuristic:v1 | gzip > 0fold-heuristic-v1.tar.gz

# -----------------------------------------------------------------------------
# Stage 1: Cython compilation. The .py sources live only in this stage and are
# discarded before the runtime stage starts. Alpine keeps the image lean; the
# compiled modules therefore target musl, which is fine because the runtime
# stage below is Alpine too.
# -----------------------------------------------------------------------------
FROM python:3.12-alpine AS builder

WORKDIR /build

# -O2 without -g keeps the compiled modules lean (the default build_ext
# flags carry debug symbols; strip is not installed on alpine).
ENV CFLAGS="-O2"

# cythonize generates C and then compiles it, so this stage needs a C
# toolchain: build-base carries gcc and the musl headers Python.h includes.
RUN apk add --no-cache build-base

RUN pip install --no-cache-dir "cython==3.0.*" "setuptools>=68"

# The bot adapter, the root advisor modules the engine imports, and the
# trimmed live-decision engine package. Only these files enter the context.
COPY bot.py .
COPY bluff.py lead_position.py risk_temperature.py .
COPY engine/ engine/

# Compile every module in place (language level 3), then drop the .py and
# generated .c files so nothing readable can be copied into stage 2.
RUN find . -name '*.py' -print0 | xargs -0 cythonize -i -3 \
    && find . \( -name '*.py' -o -name '*.c' \) -delete \
    && rm -rf build

# -----------------------------------------------------------------------------
# Stage 2: Runtime. Only the compiled .so modules + runtime deps + ENTRYPOINT.
# No .py source for the bot's strategy is present.
# -----------------------------------------------------------------------------
FROM python:3.12-alpine

# Unbuffered stdio -- without this, stdout/stderr can be lost when the
# container exits.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /bot

# The SDK from PyPI. chipzen-bot pulls in `websockets`; the engine is
# stdlib-only, so nothing else is required at runtime.
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && find /usr/local/lib/python3.12 -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /usr/local/lib/python3.12 -type d -name tests -exec rm -rf {} + 2>/dev/null || true

# Copy ONLY the compiled .so modules from the builder stage: the adapter and
# root advisors at the top level, the engine package in place. --chown folds
# ownership into the copy layer instead of duplicating it in a later chown.
COPY --from=builder --chown=10001:10001 /build/*.so /bot/
COPY --from=builder --chown=10001:10001 /build/engine/ /bot/engine/

# Run as non-root (defense in depth -- the platform also applies seccomp
# and cap-drop on top of this). Alpine uses busybox addgroup/adduser.
RUN addgroup -S -g 10001 bot && \
    adduser -S -u 10001 -G bot -h /bot -s /sbin/nologin -D bot
USER 10001

# ENTRYPOINT calls main() from the compiled bot module. The .py source isn't
# present, so the import path resolves to bot.<abi>.so via the standard
# module loader.
ENTRYPOINT ["python", "-c", "from bot import main; main()"]
