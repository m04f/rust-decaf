use std::num::NonZeroU32;

use crate::{
    hir::sym_map::{FuncSymMap, ImportSymMap, VarSymMap},
    parser::ast::*,
    span::*,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HIRLiteral {
    Int(i64),
    Bool(bool),
}

impl From<i64> for HIRLiteral {
    fn from(val: i64) -> Self {
        Self::Int(val)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Typed<T> {
    r#type: Type,
    val: T,
}

impl<T> Typed<T> {
    pub fn new(r#type: Type, val: T) -> Self {
        Self { r#type, val }
    }
    pub fn r#type(&self) -> Type {
        self.r#type
    }
    pub fn is_int(&self) -> bool {
        matches!(self.r#type, Type::Int)
    }
    pub fn is_bool(&self) -> bool {
        matches!(self.r#type, Type::Bool)
    }
    pub fn val(&self) -> &T {
        &self.val
    }
    pub fn into_val(self) -> T {
        self.val
    }
}

#[derive(Debug, Clone, Copy)]
pub enum HIRVar<'a> {
    Scalar(Typed<Span<'a>>),
    Array { arr: Typed<Span<'a>>, size: NonZeroU32 },
}

impl<'a> HIRVar<'a> {
    pub fn name(&self) -> Span<'a> {
        match self {
            Self::Scalar(ty) => ty.val,
            Self::Array { arr, .. } => arr.val,
        }
    }
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array { .. })
    }
    pub fn array_len(self) -> Option<NonZeroU32> {
        match self {
            Self::Scalar(_) => None,
            Self::Array { size, .. } => Some(size),
        }
    }
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

#[derive(Debug, Clone)]
pub enum HIRLoc<'a> {
    Scalar(Typed<Span<'a>>),
    Index {
        arr: Typed<Span<'a>>,
        size: NonZeroU32,
        index: HIRExpr<'a>,
    },
}

impl<'a> HIRLoc<'a> {
    pub fn name(&self) -> Span<'a> {
        match self {
            Self::Scalar(var) => var.val,
            Self::Index { arr, .. } => arr.val,
        }
    }
    pub fn r#type(&self) -> Type {
        match self {
            Self::Scalar(ty) => ty.r#type,
            Self::Index { arr, .. } => arr.r#type,
        }
    }
    pub fn index(var: HIRVar<'a>, index: HIRExpr<'a>) -> Self {
        assert!(var.is_array());
        match var {
            HIRVar::Scalar(_) => unreachable!(),
            HIRVar::Array { arr, size } => Self::Index {
                arr: Typed {
                    r#type: arr.r#type,
                    val: arr.val,
                },
                size,
                index,
            },
        }
    }
}

impl HIRLiteral {
    pub fn int(self) -> Option<i64> {
        match self {
            Self::Int(val) => Some(val),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
}

impl From<ArithOp> for Op {
    fn from(value: ArithOp) -> Self {
        match value {
            ArithOp::Add => Op::Add,
            ArithOp::Sub => Op::Sub,
            ArithOp::Mul => Op::Mul,
            ArithOp::Div => Op::Div,
            ArithOp::Mod => Op::Mod,
        }
    }
}

impl TryFrom<Op> for ArithOp {
    type Error = ();
    fn try_from(op: Op) -> Result<Self, Self::Error> {
        match op {
            Op::Add => Ok(ArithOp::Add),
            Op::Sub => Ok(ArithOp::Sub),
            Op::Mul => Ok(ArithOp::Mul),
            Op::Div => Ok(ArithOp::Div),
            Op::Mod => Ok(ArithOp::Mod),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum RelOp {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

impl From<RelOp> for Op {
    fn from(value: RelOp) -> Self {
        match value {
            RelOp::Less => Op::Less,
            RelOp::LessEqual => Op::LessEqual,
            RelOp::Greater => Op::Greater,
            RelOp::GreaterEqual => Op::GreaterEqual,
        }
    }
}

impl TryFrom<Op> for RelOp {
    type Error = ();
    fn try_from(op: Op) -> Result<Self, Self::Error> {
        match op {
            Op::Less => Ok(RelOp::Less),
            Op::LessEqual => Ok(RelOp::LessEqual),
            Op::Greater => Ok(RelOp::Greater),
            Op::GreaterEqual => Ok(RelOp::GreaterEqual),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EqOp {
    Equal,
    NotEqual,
}

impl From<EqOp> for Op {
    fn from(value: EqOp) -> Self {
        match value {
            EqOp::Equal => Op::Equal,
            EqOp::NotEqual => Op::NotEqual,
        }
    }
}
impl TryFrom<Op> for EqOp {
    type Error = ();
    fn try_from(op: Op) -> Result<Self, Self::Error> {
        match op {
            Op::Equal => Ok(EqOp::Equal),
            Op::NotEqual => Ok(EqOp::NotEqual),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CondOp {
    And,
    Or,
}

impl From<CondOp> for Op {
    fn from(value: CondOp) -> Self {
        match value {
            CondOp::And => Op::And,
            CondOp::Or => Op::Or,
        }
    }
}

impl TryFrom<Op> for CondOp {
    type Error = ();
    fn try_from(op: Op) -> Result<Self, Self::Error> {
        match op {
            Op::And => Ok(CondOp::And),
            Op::Or => Ok(CondOp::Or),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HIRExpr<'a> {
    Len(NonZeroU32),
    Not(Box<HIRExpr<'a>>),
    Neg(Box<HIRExpr<'a>>),
    Ter {
        cond: Box<HIRExpr<'a>>,
        yes: Box<HIRExpr<'a>>,
        no: Box<HIRExpr<'a>>,
    },
    Call(HIRCall<'a>),
    Loc(Box<HIRLoc<'a>>),
    Literal(HIRLiteral),
    Arith {
        op: ArithOp,
        lhs: Box<HIRExpr<'a>>,
        rhs: Box<HIRExpr<'a>>,
    },
    Rel {
        op: RelOp,
        lhs: Box<HIRExpr<'a>>,
        rhs: Box<HIRExpr<'a>>,
    },
    Eq {
        op: EqOp,
        lhs: Box<HIRExpr<'a>>,
        rhs: Box<HIRExpr<'a>>,
    },

    Cond {
        op: CondOp,
        lhs: Box<HIRExpr<'a>>,
        rhs: Box<HIRExpr<'a>>,
    },
}

impl<'a> From<HIRLoc<'a>> for HIRExpr<'a> {
    fn from(value: HIRLoc<'a>) -> Self {
        Self::Loc(Box::new(value))
    }
}

impl<'a> From<HIRLiteral> for HIRExpr<'a> {
    fn from(value: HIRLiteral) -> Self {
        Self::Literal(value)
    }
}

impl<'a> From<HIRCall<'a>> for HIRExpr<'a> {
    fn from(value: HIRCall<'a>) -> Self {
        Self::Call(value)
    }
}

impl<'a> HIRExpr<'a> {
    pub fn new_not(self) -> Self {
        assert!(self.r#type() == Type::Bool);
        Self::Not(Box::new(self))
    }

    pub fn new_neg(self) -> Self {
        assert!(self.r#type() == Type::Int);
        Self::Neg(Box::new(self))
    }

    pub fn nested(self) -> Self {
        self
    }

    pub fn is_boolean(&self) -> bool {
        self.r#type() == Type::Bool
    }

    pub fn is_int(&self) -> bool {
        self.r#type() == Type::Int
    }

    pub fn r#type(&self) -> Type {
        use HIRExpr::*;
        use HIRLiteral::*;
        match self {
            Len(_) | Neg(_) | Arith { .. } | Literal(Int(_)) => Type::Int,
            Cond { .. } | Eq { .. } | Rel { .. } | Not(..) | Literal(Bool(_)) => Type::Bool,
            Ter { yes, .. } => yes.r#type(),
            Loc(loc) => loc.r#type(),
            Call(call) => call.return_type().unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum HIRCall<'a> {
    Extern {
        name: Span<'a>,
        args: Vec<ExternArg<'a>>,
    },
    Decaf {
        name: Span<'a>,
        ret: Option<Type>,
        args: Vec<HIRExpr<'a>>,
    },
}

impl<'a> HIRCall<'a> {
    fn return_type(&self) -> Option<Type> {
        match self {
            Self::Extern { .. } => Some(Type::Int),
            Self::Decaf { ret, .. } => *ret,
        }
    }
    pub fn new_extern(name: Span<'a>, args: Vec<ExternArg<'a>>) -> Self {
        Self::Extern { name, args }
    }
    pub fn new_decaf(name: Span<'a>, ret: Option<Type>, args: Vec<HIRExpr<'a>>) -> Self {
        Self::Decaf { name, ret, args }
    }
}

#[derive(Debug, Clone)]
pub enum HIRStmt<'a> {
    Assign(HIRAssign<'a>),
    Expr(HIRExpr<'a>),
    Return(Option<HIRExpr<'a>>),
    Break,
    Continue,
    If {
        cond: HIRExpr<'a>,
        yes: Box<HIRBlock<'a>>,
        no: Box<HIRBlock<'a>>,
    },
    While {
        cond: HIRExpr<'a>,
        body: Box<HIRBlock<'a>>,
    },
    For {
        init: HIRAssign<'a>,
        cond: HIRExpr<'a>,
        update: HIRAssign<'a>,
        body: Box<HIRBlock<'a>>,
    },
}

impl<'a> From<HIRAssign<'a>> for HIRStmt<'a> {
    fn from(value: HIRAssign<'a>) -> Self {
        Self::Assign(value)
    }
}

#[derive(Debug, Clone)]
pub struct HIRFunction<'a> {
    pub name: Span<'a>,
    pub body: HIRBlock<'a>,
    pub args: VarSymMap<'a>,
    // redundent but easy...
    pub args_sorted: Vec<Span<'a>>,
    pub ret: Option<Type>,
}

impl<'a> HIRFunction<'a> {
    pub fn new(
        name: PIdentifier<'a>,
        body: HIRBlock<'a>,
        args: VarSymMap<'a>,
        args_sorted: Vec<Span<'a>>,
        ret: Option<Type>,
    ) -> Self {
        Self {
            name: name.span(),
            body,
            args,
            args_sorted,
            ret,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HIRBlock<'a> {
    pub decls: VarSymMap<'a>,
    pub stmts: Vec<HIRStmt<'a>>,
}

impl<'a> HIRBlock<'a> {
    pub fn decls(&self) -> &VarSymMap<'a> {
        &self.decls
    }
}

#[derive(Debug, Clone)]
pub enum FunctionSig<'a> {
    Extern(Span<'a>),
    Decl {
        name: Span<'a>,
        arg_types: Vec<Type>,
        ty: Option<Type>,
    },
}

impl<'a> FunctionSig<'a> {
    pub fn name(&self) -> Span<'a> {
        match self {
            Self::Extern(name) => *name,
            Self::Decl { name, .. } => *name,
        }
    }
    pub fn get(func: &PFunction<'a>) -> Self {
        Self::Decl {
            name: func.name.span(),
            arg_types: func.args.iter().map(|arg| arg.r#type()).collect(),
            ty: func.ret,
        }
    }
    pub(super) fn from_pimport(import: &PImport<'a>) -> Self {
        Self::Extern(import.name().span())
    }
}

#[derive(Debug, Clone)]
pub enum ExternArg<'a> {
    String(&'a str),
    Array(Span<'a>),
    Expr(HIRExpr<'a>),
}

impl<'a> From<PString<'a>> for ExternArg<'a> {
    fn from(value: PString<'a>) -> Self {
        Self::String(value.span().as_str())
    }
}

#[derive(Debug, Clone)]
pub struct HIRRoot<'a> {
    pub globals: VarSymMap<'a>,
    pub functions: FuncSymMap<'a>,
    pub imports: ImportSymMap<'a>,
}

#[derive(Debug, Clone)]
pub struct HIRAssign<'a> {
    pub lhs: HIRLoc<'a>,
    pub rhs: HIRExpr<'a>,
}
