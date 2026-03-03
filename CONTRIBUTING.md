# Contributing to nQPU

Thank you for your interest in contributing to nQPU!

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/nqpu.git
cd nqpu

# Python development
cd sdk/python
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Rust development (optional)
cd sdk/rust
cargo build
cargo test
```

## Code Style

- Python: Follow PEP 8, use `black` for formatting
- Rust: Follow standard Rust conventions, use `cargo fmt`

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Areas for Contribution

- **Core**: New quantum gates, optimization passes
- **Chemistry**: Molecular fingerprints, drug discovery algorithms
- **Biology**: Quantum biology simulations
- **Documentation**: Tutorials, API docs
- **Examples**: Jupyter notebooks, demos

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
