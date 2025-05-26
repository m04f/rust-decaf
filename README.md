# rust-decaf

This repository is an implementation of the Decaf programming language, created as part of MIT's 6.035 (Computer Language Engineering) course. The Decaf language and course materials can be found at the [6.035 course website](http://6.035.scripts.mit.edu/fa18/).

## About

Decaf is a simplified Java-like language designed for educational purposes. This project represents a Rust-based implementation of the Decaf language, including its parser, semantic analysis, and code generation components.

## Features

- Lexical analysis (tokenization)
- Parsing of Decaf source code
- Semantic analysis (type checking, symbol tables, etc.)
- Intermediate representation and code generation

## Getting Started

### Prerequisites

- Rust toolchain (latest stable recommended). Install via [rustup](https://rustup.rs/).

### Building

Clone the repository and build with Cargo:

```bash
git clone https://github.com/m04f/rust-decaf.git
cd rust-decaf
cargo build
```

### Running

You can run the Decaf compiler with:

```bash
cargo run -- decafcc
```

## Project Structure

- `src/` - Rust source code for the Decaf implementation
- `tests/` - Test cases for language features
- Other files and directories follow standard Rust project layout

## References

- [MIT 6.035: Computer Language Engineering](http://6.035.scripts.mit.edu/fa18/)
- [Decaf Language Specification]([http://6.035.scripts.mit.edu/fa18/decaf.pdf](http://6.035.scripts.mit.edu/fa18/handout-pdfs/01-decaf-spec.pdf))

---

If you are interested in a more refined or alternative implementation, see the rewrite at [decafc](https://github.com/mostafa-khaled775/decafc).
