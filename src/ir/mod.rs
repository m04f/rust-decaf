//! TODO: this module requries alot of refactoring.

mod basicblock;
mod cfg_construction;
mod dot;
mod graph;
mod instruction;
#[cfg(test)]
mod test;
pub use basicblock::*;
pub use dot::*;
pub use graph::*;
pub use instruction::*;
