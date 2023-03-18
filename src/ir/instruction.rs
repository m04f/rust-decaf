use std::fmt::{Debug, Display};

use crate::{hir::*, parser, span::*};

pub type Op = parser::ast::Op;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Symbol<'b>(pub(super) &'b str, pub(super) u16);

impl<'a> Symbol<'a> {
    pub fn name(&self) -> &'a str {
        self.0
    }

    pub fn global(name: &'a str) -> Self {
        Self(name, 0)
    }
}

impl Display for Symbol<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", self.0, self.1)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Immediate {
    Int(i64),
    Bool(bool),
}

impl From<Immediate> for i64 {
    fn from(value: Immediate) -> Self {
        match value {
            Immediate::Int(i) => i,
            Immediate::Bool(b) => b as i64,
        }
    }
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

#[derive(Debug, Clone, Copy)]
pub enum Source<'b> {
    Immediate(Immediate),
    Symbol(Symbol<'b>),
    Offset(Symbol<'b>, Reg),
    Reg(Reg),
    Sc(Sc),
}

impl From<Sc> for Source<'_> {
    fn from(value: Sc) -> Self {
        Self::Sc(value)
    }
}

impl<'a> From<Symbol<'a>> for Source<'a> {
    fn from(value: Symbol<'a>) -> Self {
        Self::Symbol(value)
    }
}

impl From<HIRLiteral> for Source<'_> {
    fn from(value: HIRLiteral) -> Self {
        Self::Immediate(value.into())
    }
}

impl From<bool> for Source<'_> {
    fn from(value: bool) -> Self {
        Self::Immediate(Immediate::Bool(value))
    }
}

impl<'b> Source<'b> {
    pub fn to_dist(&self) -> Option<Dest<'b>> {
        match *self {
            Self::Reg(reg) => Some(Dest::Reg(reg)),
            Self::Symbol(s) => Some(Dest::Symbol(s)),
            Self::Offset(s, r) => Some(Dest::Offset(s, r)),
            Self::Sc(sc) => Some(Dest::Sc(sc)),
            Self::Immediate(_) => None,
        }
    }

    pub fn immediate(&self) -> Option<Immediate> {
        match self {
            Self::Immediate(i) => Some(*i),
            _ => None,
        }
    }
}

impl Display for Source<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Immediate(i) => write!(f, "{i}"),
            Self::Symbol(s) => write!(f, "%{s}"),
            Self::Offset(s, r) => write!(f, "%{s}[{r}]"),
            Self::Sc(sc) => write!(f, "{sc}"),
            Self::Reg(r) => write!(f, "{r}"),
        }
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

impl From<Immediate> for Source<'_> {
    fn from(value: Immediate) -> Self {
        Self::Immediate(value)
    }
}

impl From<Reg> for Source<'_> {
    fn from(value: Reg) -> Self {
        Self::Reg(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Dest<'b> {
    Symbol(Symbol<'b>),
    Offset(Symbol<'b>, Reg),
    Reg(Reg),
    Sc(Sc),
}

impl From<Sc> for Dest<'_> {
    fn from(value: Sc) -> Self {
        Self::Sc(value)
    }
}

impl<'a> From<Symbol<'a>> for Dest<'a> {
    fn from(value: Symbol<'a>) -> Self {
        Self::Symbol(value)
    }
}

impl Display for Dest<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Symbol(s) => write!(f, "%{s}"),
            Self::Offset(s, r) => write!(f, "%{s}[{r}]"),
            Self::Reg(r) => write!(f, "{r}"),
            Self::Sc(sc) => write!(f, "{sc}"),
        }
    }
}

impl From<Reg> for Dest<'_> {
    fn from(value: Reg) -> Self {
        Self::Reg(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Unary {
    Not,
    Neg,
}

#[derive(Clone)]
pub enum IRExternArg<'b> {
    Source(Source<'b>),
    String(&'b str),
}

impl Debug for IRExternArg<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(s) => write!(f, "{s}"),
            Self::String(_) => write!(f, "const string"),
        }
    }
}

#[rustfmt::skip]
#[derive(Clone)]
pub enum Instruction<'b> {
    AllocArray { name: Symbol<'b>, size: u64 },
    AllocScalar { name: Symbol<'b> },
    Op2 { dest: Dest<'b>, lhs: Source<'b>, rhs: Source<'b>, op: Op },
    Unary { dest: Dest<'b>, source: Source<'b>, op: Unary },
    Move { dest: Dest<'b>, source: Source<'b> },
    Select { dest: Dest<'b>, cond: Source<'b>, yes: Source<'b>, no: Source<'b> },
    Exit(i8),
    VoidCall { symbol: &'b str, args: Vec<Reg> },
    Call { dest: Dest<'b>, symbol: &'b str, args: Vec<Reg> },
    ExternCall { dest: Dest<'b>, symbol: &'b str, args: Vec<IRExternArg<'b>> },
    Return { value: Source<'b> },
    VoidReturn,
}

impl Debug for Instruction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Op2 { dest, lhs, rhs, op } => write!(f, "{dest} = {op:?} {lhs} {rhs}"),
            Self::AllocArray { name, size } => write!(f, "alloc {name}[{size}]"),
            Self::AllocScalar { name } => write!(f, "alloc {name}"),
            Self::Move { dest, source } => write!(f, "{dest} = {source}"),
            Self::Unary { dest, source, op } => write!(f, "{dest} = {op:?} {source}"),
            Self::Exit(code) => write!(f, "exit {code}"),
            Self::Select {
                dest,
                cond,
                yes,
                no,
            } => write!(f, "{} = select {} {} {}", dest, cond, yes, no),
            Self::VoidCall { symbol, args } => write!(f, "void call {} {:?}", symbol, args),
            Self::Call { dest, symbol, args } => write!(f, "{} = call {} {:?}", dest, symbol, args),
            Self::ExternCall { dest, symbol, args } => {
                write!(f, "{} = extern call {} {:?}", dest, symbol, args)
            }
            Self::Return { value } => write!(f, "return {}", value),
            Self::VoidReturn => write!(f, "ret"),
        }
    }
}

impl<'b> Instruction<'b> {
    pub fn new_load(reg: Reg, symbol: impl Into<Source<'b>>) -> Self {
        Self::Move {
            dest: reg.into(),
            source: symbol.into(),
        }
    }

    pub fn new_load_offset(reg: Reg, symbol: Symbol<'b>, offset: Reg) -> Self {
        Self::Move {
            dest: reg.into(),
            source: Source::Offset(symbol, offset),
        }
    }

    pub fn new_store(symbol: impl Into<Dest<'b>>, source: impl Into<Source<'b>>) -> Self {
        Self::Move {
            dest: symbol.into(),
            source: source.into(),
        }
    }

    pub fn new_store_offset(symbol: Symbol<'b>, offset: Reg, source: Reg) -> Self {
        Self::Move {
            dest: Dest::Offset(symbol, offset),
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

    pub fn new_void_call(symbol: Span<'b>, args: Vec<Reg>) -> Self {
        Self::VoidCall {
            symbol: symbol.as_str(),
            args,
        }
    }
    pub fn new_void_ret() -> Self {
        Self::VoidReturn
    }
    pub fn new_ret_call(reg: Reg, symbol: Span<'b>, args: Vec<Reg>) -> Self {
        Self::Call {
            dest: reg.into(),
            symbol: symbol.as_str(),
            args,
        }
    }
    pub fn new_extern_call(reg: Reg, symbol: Span<'b>, args: Vec<IRExternArg<'b>>) -> Self {
        Self::ExternCall {
            dest: reg.into(),
            symbol: symbol.as_str(),
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
            lhs: lhs.into(),
            rhs: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_eq(dest: Reg, lhs: Reg, op: EqOp, rhs: Reg) -> Self {
        Self::Op2 {
            dest: dest.into(),
            lhs: lhs.into(),
            rhs: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_cond(dest: Reg, lhs: Reg, op: CondOp, rhs: Reg) -> Self {
        Self::Op2 {
            dest: dest.into(),
            lhs: lhs.into(),
            rhs: rhs.into(),
            op: op.into(),
        }
    }
    pub fn new_rel(dest: Reg, lhs: Reg, op: RelOp, rhs: Reg) -> Self {
        Self::Op2 {
            dest: dest.into(),
            lhs: lhs.into(),
            rhs: rhs.into(),
            op: op.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Reg(u32);

impl Reg {
    pub(super) fn new(num: u32) -> Self {
        Self(num)
    }

    pub fn num(&self) -> u32 {
        self.0
    }
}

/// A short circuit register.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Sc(u32);

impl Sc {
    pub(super) fn new(num: u32) -> Self {
        Self(num)
    }

    pub fn num(&self) -> u32 {
        self.0
    }
}

impl Display for Reg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%{}", self.0)
    }
}

impl Display for Sc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "%sc{}", self.0)
    }
}
