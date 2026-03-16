# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | Yes                |

## Reporting a Vulnerability

The nQPU team takes security vulnerabilities seriously. We appreciate your efforts
to responsibly disclose any issues you find.

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to:

**security@entropy.ai**

### What to Include

- A description of the vulnerability and its potential impact
- Steps to reproduce the issue or a proof-of-concept
- The affected component(s) (Rust SDK, Python SDK, specific module)
- Any suggested remediation if you have one

### Response Timeline

- **Acknowledgment**: Within 48 hours of your report
- **Initial assessment**: Within 5 business days
- **Resolution target**: Within 30 days for critical issues, 90 days for others

### What to Expect

- We will confirm receipt of your report and provide a tracking identifier
- We will investigate and keep you informed of our progress
- We will credit you in the advisory (unless you prefer to remain anonymous)
- We will not take legal action against researchers acting in good faith

### Scope

This policy applies to all code in the nQPU repository, including:

- Rust SDK (`sdk/rust/`)
- Python SDK (`sdk/python/`)
- Quantum key distribution and cryptographic modules
- Random number generation modules

### Out of Scope

- Theoretical quantum computing vulnerabilities (e.g., Shor's algorithm implications)
- Issues in third-party dependencies (please report to those projects directly)
- Denial-of-service attacks against CI/CD infrastructure

## Preferred Languages

We accept vulnerability reports in English.
