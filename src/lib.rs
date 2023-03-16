#![feature(if_let_guard)]
#![feature(build_hasher_simple_hash_one)]

pub mod error;
pub mod lexer;
pub mod log;
pub mod parser;
pub mod hir;
pub mod span;
pub mod ir;

/// a module that comtains any optimizations made to the ir. (currently...)
pub mod optimizations;

/// codegen module (for x86_64 only).
pub mod codegen;
