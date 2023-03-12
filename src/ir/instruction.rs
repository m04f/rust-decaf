use std::fmt::{Debug, Display};

// TODO: define an enum here instead.
use crate::parser::ast::Op;

use crate::hir::*;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum Reg {
    Global(usize),
    Local(usize),
}

#[derive(Debug, Clone, Copy)]
pub struct FunctionSymbol(usize);

#[derive(Debug, Clone, Copy)]
pub enum Immediate {
    Int(i64),
    Bool(bool),
}

impl From<i64> for Immediate {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

#[derive(Debug, Clone)]
pub enum Source {
    Immediate(Immediate),
    Index { base: Reg, offset: Reg },
    Reg(Reg),
    UnInit,
}

impl From<HIRLiteral> for Immediate {
    fn from(value: HIRLiteral) -> Self {
        match value {
            HIRLiteral::Int(i) => Self::Int(i),
            HIRLiteral::Bool(b) => Self::Bool(b),
        }
    }
}

impl From<Immediate> for Source {
    fn from(value: Immediate) -> Self {
        Self::Immediate(value)
    }
}

impl From<Reg> for Source {
    fn from(value: Reg) -> Self {
        Self::Reg(value)
    }
}

#[derive(Debug, Clone)]
pub enum Dest {
    Index { base: Reg, offset: Reg },
    Reg(Reg),
}

impl From<Reg> for Dest {
    fn from(value: Reg) -> Self {
        Self::Reg(value)
    }
}

#[derive(Debug, Clone)]
pub enum Unary {
    Not,
    Neg,
}

#[rustfmt::skip]
#[derive(Clone, Debug)]
pub enum Instruction {
    BinOp { dest: Dest, source1: Source, source2: Source, op: Op },
    Add { dest: Dest, lhs: Source, rhs: Source },
    Sub { dest: Dest, lhs: Source, rhs: Source },
    Unary { dest: Dest, source: Source, op: Unary },
    Select { dest: Dest, cond: Source, yes: Source, no: Source },
    Move { dest: Dest, source: Source },
    Phi { dest: Dest, sources: Vec<Source> },
    ReturnGuard,
    VoidCall { symbol: FunctionSymbol, args: Vec<Reg> },
    Call { dest: Dest, symbol: FunctionSymbol, args: Vec<Reg> },
    Return { value: Source },
    VoidReturn,
}

impl Instruction {
    pub fn new_return(reg: Reg) -> Self {
        Self::Return { value: reg.into() }
    }
    pub fn new_select(reg: Reg, cond: Reg, yes: Reg, no: Reg) -> Self {
        Self::Select {
            dest: reg.into(),
            cond: cond.into(),
            yes: yes.into(),
            no: no.into(),
        }
    }

    pub fn new_void_call(symbol: impl Into<Symbol>, args: Vec<Reg>) -> Self {
        Self::VoidCall {
            symbol: symbol.into(),
            args,
        }
    }
    pub fn new_void_ret() -> Self {
        Self::VoidReturn
    }
    pub fn new_ret_call(reg: Reg, symbol: impl Into<Symbol>, args: Vec<Reg>) -> Self {
        Self::Call {
            dest: reg.into(),
            symbol: symbol.into(),
            args,
        }
    }
    pub fn new_neg(dest: Reg, source: Reg) -> Self {
        Self::Unary {
            dest: dest.into(),
            source: source.into(),
            op: Unary::Neg,
        }
    }
    pub fn new_not(dest: Reg, source: Reg) -> Self {
        Self::Unary {
            dest: dest.into(),
            source: source.into(),
            op: Unary::Not,
        }
    }
    pub fn new_arith(dest: Reg, lhs: Reg, op: ArithOp, rhs: Reg) -> Self {
        Self::BinOp {
            dest: dest.into(),
            source1: lhs.into(),
            source2: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_eq(dest: Reg, lhs: Reg, op: EqOp, rhs: Reg) -> Self {
        Self::BinOp {
            dest: dest.into(),
            source1: lhs.into(),
            source2: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_cond(dest: Reg, lhs: Reg, op: CondOp, rhs: Reg) -> Self {
        Self::BinOp {
            dest: dest.into(),
            source1: lhs.into(),
            source2: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_rel(dest: Reg, lhs: Reg, op: RelOp, rhs: Reg) -> Self {
        Self::BinOp {
            dest: dest.into(),
            source1: lhs.into(),
            source2: rhs.into(),
            op: op.into(),
        }
    }
}
