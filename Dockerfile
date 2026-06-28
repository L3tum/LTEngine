# syntax=docker/dockerfile:1

# Build with musl for a fully static binary (no dynamic libc dependencies)
FROM rust:1.96 AS builder

ENV PATH="/root/.cargo/bin:${PATH}"

# Install musl target and the musl-gcc compiler needed by ring
RUN rustup target add x86_64-unknown-linux-musl \
    && apt-get update && apt-get install -y --no-install-recommends musl-tools \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source
COPY . .

# Build with caching and musl toolchain
RUN --mount=type=cache,id=ltengine-cargo-registry,target=/root/.cargo/registry \
    --mount=type=cache,id=ltengine-target,target=/build/target \
    cargo build --target x86_64-unknown-linux-musl --release -p ltengine && \
    cp target/x86_64-unknown-linux-musl/release/ltengine /ltengine

# Runtime: completely static binary with no external dependencies
FROM scratch

COPY --from=builder /ltengine /ltengine

ENV LTE_BACKEND_HOST=localhost:11434
ENV LTE_MODEL=gemma3-4b

EXPOSE 5050

CMD ["ltengine"]
