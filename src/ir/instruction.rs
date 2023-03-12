use std::fmt::{Debug, Display};

use crate::{hir::*, parser, span::*};

pub type Op = parser::ast::Op;

pub type Symbol = String;

#[derive(Debug, Clone, Copy)]
pub enum Immediate {
    Int(i64),
    Bool(bool),
}

impl Display for Immediate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Int(i) => i.to_string(),
                Self::Bool(b) => b.to_string(),
            }
        )
    }
}

impl From<i64> for Immediate {
    fn from(value: i64) -> Self {
        Self::Int(value)
    }
}

#[derive(Debug, Clone)]
pub enum Source {
    Immediate(Immediate),
    Symbol(Symbol),
    Offset(Symbol, Reg),
    Reg(Reg),
}

impl Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Immediate(i) => write!(f, "{i}"),
            Self::Symbol(s) => write!(f, "%{s}"),
            Self::Offset(s, r) => write!(f, "%{s}[{r}]"),
            Self::Reg(r) => write!(f, "{r}"),
        }
    }
}

impl From<Span<'_>> for Symbol {
    fn from(value: Span<'_>) -> Self {
        value.to_string()
    }
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

impl From<Symbol> for Source {
    fn from(value: Symbol) -> Self {
        Self::Symbol(value)
    }
}

#[derive(Debug, Clone)]
pub enum Dest {
    Symbol(Symbol),
    Offset(Symbol, Reg),
    Reg(Reg),
}

impl Display for Dest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Symbol(s) => write!(f, "%{s}"),
            Self::Offset(s, r) => write!(f, "%{s}[{r}]"),
            Self::Reg(r) => write!(f, "{r}"),
        }
    }
}

impl Dest {
    fn reg(self) -> Option<Reg> {
        match self {
            Self::Reg(reg) => Some(reg),
            Self::Symbol(_) => None,
            Self::Offset(_, _) => None,
        }
    }
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
#[derive(Clone)]
pub enum Instruction {
    Op2 { dest: Dest, source1: Source, source2: Source, op: Op },
    Unary { dest: Dest, source: Source, op: Unary },
    Load { dest: Dest, source: Source },
    Store { dest: Dest, source: Source },
    Select { dest: Dest, cond: Source, yes: Source, no: Source },
    BoundGuard { value: Source, bound: u64 },
    ReturnGuard,
    StackAlloc { symbol: Symbol },
    VoidCall { symbol: Symbol, args: Vec<Reg> },
    Call { dest: Dest, symbol: Symbol, args: Vec<Reg> },
    Return { value: Source },
    VoidReturn,
}

impl Debug for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Op2 {
                dest,
                source1,
                source2,
                op,
            } => write!(f, "{dest} = {op:?} {source1} {source2}"),
            Self::Load { dest, source } => write!(f, "{dest} = load {source}"),
            Self::Store { dest, source } => write!(f, "store {dest} {source}"),
            Self::Unary { dest, source, op } => write!(f, "{dest} = {op:?} {source}"),
            Self::Select {
                dest,
                cond,
                yes,
                no,
            } => write!(f, "{} = select {} {} {}", dest, cond, yes, no),
            Self::BoundGuard { value, bound } => write!(f, "bound_guard {} {}", value, bound),
            Self::ReturnGuard => write!(f, "return guard"),
            Self::StackAlloc { symbol } => write!(f, "stack_alloc {}", symbol),
            Self::VoidCall { symbol, args } => write!(f, "void call {} {:?}", symbol, args),
            Self::Call { dest, symbol, args } => write!(f, "{} = call {} {:?}", dest, symbol, args),
            Self::Return { value } => write!(f, "return {}", value),
            Self::VoidReturn => write!(f, "ret"),
        }
    }
}

impl Instruction {
    pub fn new_load(reg: Reg, symbol: impl Into<Symbol>) -> Self {
        Self::Load {
            dest: reg.into(),
            source: symbol.into().into(),
        }
    }
    pub fn new_load_imm(reg: Reg, immediate: impl Into<Immediate>) -> Self {
        Self::Load {
            dest: reg.into(),
            source: immediate.into().into(),
        }
    }
    pub fn new_load_offset(reg: Reg, symbol: impl Into<Symbol>, offset: Reg, _bound: u64) -> Self {
        Self::Load {
            dest: reg.into(),
            source: Source::Offset(symbol.into(), offset),
        }
    }
    pub fn new_bound_check(reg: Reg, bound: u64) -> Self {
        Self::BoundGuard {
            value: reg.into(),
            bound,
        }
    }
    pub fn new_store(symbol: impl Into<Symbol>, source: Reg) -> Self {
        Self::Store {
            dest: Dest::Symbol(symbol.into()),
            source: source.into(),
        }
    }
    pub fn new_store_offset(symbol: impl Into<Symbol>, offset: Reg, source: Reg, _bound: u64) -> Self {
        Self::Store {
            dest: Dest::Offset(symbol.into(), offset),
            source: source.into(),
        }
    }
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
    pub fn new_stack_alloc(symbol: impl Into<Symbol>) -> Self {
        Self::StackAlloc {
            symbol: symbol.into(),
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
        Self::Op2 {
            dest: dest.into(),
            source1: lhs.into(),
            source2: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_eq(dest: Reg, lhs: Reg, op: EqOp, rhs: Reg) -> Self {
        Self::Op2 {
            dest: dest.into(),
            source1: lhs.into(),
            source2: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_cond(dest: Reg, lhs: Reg, op: CondOp, rhs: Reg) -> Self {
        Self::Op2 {
            dest: dest.into(),
            source1: lhs.into(),
            source2: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_rel(dest: Reg, lhs: Reg, op: RelOp, rhs: Reg) -> Self {
        Self::Op2 {
            dest: dest.into(),
            source1: lhs.into(),
            source2: rhs.into(),
            op: op.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Reg(usize);

impl Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl Reg {
    pub fn new(id: usize) -> Self {
        Self(id)
    }
}
