# LTEngine — Quality & Development Makefile
# Run with `make` or `make <target>`

.PHONY: check fmt clippy test miri deny all

# Quick pre-commit checks
check:
	cargo check --all-targets --all-features

fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

clippy:
	cargo clippy --all-targets --all-features -- -D warnings

test:
	cargo test --all-targets

# MIRI — detects undefined behavior (slow, needs nightly toolchain + miri component)
# All tests use mocks — no real network calls needed, so isolation is safe
miri:
	cargo +nightly miri test

# cargo-deny — license, security, banned crate checks
deny:
	cargo deny check licenses advisories ban

# Full pre-merge check suite
all: check fmt-check clippy test deny

# Build for release (static musl binary)
build-release:
	cargo build --release

# Build docs
docs:
	cargo doc --open --no-deps
