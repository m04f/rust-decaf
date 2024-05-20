use crate::{
    hir::sym_map::{FuncSymMap, ImportSymMap, VarSymMap},
    parser::ast::*, span::Span,
};

pub type Identifier = String;

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
    pub fn val(&self) -> &T {
        &self.val
    }
    pub fn into_val(self) -> T {
        self.val
    }
}

#[derive(Debug, Clone)]
pub enum HIRVar {
    Scalar(Typed<Identifier>),
    Array { arr: Typed<Identifier>, size: u64 },
}

impl HIRVar {
    pub fn name(&self) -> &str {
        match self {
            Self::Scalar(ty) => &ty.val,
            Self::Array { arr, .. } => &arr.val,
        }
    }
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array { .. })
    }
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar(_))
    }
}

#[derive(Debug, Clone)]
pub enum HIRLoc {
    Scalar(Typed<Identifier>),
    Index {
        arr: Typed<Identifier>,
        size: u64,
        index: HIRExpr,
    },
}

impl HIRLoc {
    pub fn name(&self) -> &str {
        match self {
            Self::Scalar(var) => &var.val,
            Self::Index { arr, .. } => &arr.val,
        }
    }
    pub fn r#type(&self) -> Type {
        match self {
            Self::Scalar(ty) => ty.r#type,
            Self::Index { arr, .. } => arr.r#type,
        }
    }
    pub fn index(var: HIRVar, index: HIRExpr) -> Self {
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
pub enum HIRExpr {
    Len(u64),
    Not(Box<HIRExpr>),
    Neg(Box<HIRExpr>),
    Ter {
        cond: Box<HIRExpr>,
        yes: Box<HIRExpr>,
        no: Box<HIRExpr>,
    },
    Call(HIRCall),
    Loc(Box<HIRLoc>),
    Literal(HIRLiteral),
    Arith {
        op: ArithOp,
        lhs: Box<HIRExpr>,
        rhs: Box<HIRExpr>,
    },
    Rel {
        op: RelOp,
        lhs: Box<HIRExpr>,
        rhs: Box<HIRExpr>,
    },
    Eq {
        op: EqOp,
        lhs: Box<HIRExpr>,
        rhs: Box<HIRExpr>,
    },

    Cond {
        op: CondOp,
        lhs: Box<HIRExpr>,
        rhs: Box<HIRExpr>,
    },
}

impl From<HIRLoc> for HIRExpr {
    fn from(value: HIRLoc) -> Self {
        Self::Loc(Box::new(value))
    }
}

impl From<HIRLiteral> for HIRExpr {
    fn from(value: HIRLiteral) -> Self {
        Self::Literal(value)
    }
}

impl From<HIRCall> for HIRExpr {
    fn from(value: HIRCall) -> Self {
        Self::Call(value)
    }
}

impl HIRExpr {
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
pub enum HIRCall {
    Extern {
        name: String,
        args: Vec<ExternArg>,
    },
    Decaf {
        name: String,
        ret: Option<Type>,
        args: Vec<HIRExpr>,
    },
}

impl HIRCall {
    fn return_type(&self) -> Option<Type> {
        match self {
            Self::Extern { .. } => Some(Type::Int),
            Self::Decaf { ret, .. } => *ret,
        }
    }
    pub fn new_extern(name: String, args: Vec<ExternArg>) -> Self {
        Self::Extern { name, args }
    }
    pub fn new_decaf(name: String, ret: Option<Type>, args: Vec<HIRExpr>) -> Self {
        Self::Decaf { name, ret, args }
    }
}

#[derive(Debug, Clone)]
pub enum HIRStmt {
    Assign(HIRAssign),
    Expr(HIRExpr),
    Return(Option<HIRExpr>),
    Break,
    Continue,
    If {
        cond: HIRExpr,
        yes: Box<HIRBlock>,
        no: Box<HIRBlock>,
    },
    While {
        cond: HIRExpr,
        body: Box<HIRBlock>,
    },
    For {
        init: HIRAssign,
        cond: HIRExpr,
        update: HIRAssign,
        body: Box<HIRBlock>,
    },
}

#[derive(Debug, Clone)]
pub struct HIRFunction {
    pub name: String,
    pub body: HIRBlock,
    pub args: VarSymMap,
    pub ret: Option<Type>,
}

impl HIRFunction {
    pub fn new(
        name: Span,
        body: HIRBlock,
        args: VarSymMap,
        ret: Option<Type>,
    ) -> Self {
        Self {
            name: name.to_string(),
            body,
            args,
            ret,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HIRBlock {
    pub decls: VarSymMap,
    pub stmts: Vec<HIRStmt>,
}

impl HIRBlock {
    pub fn decls(&self) -> &VarSymMap {
        &self.decls
    }
}

#[derive(Debug, Clone)]
pub enum FunctionSig {
    Extern(String),
    Decl {
        name: String,
        arg_types: Vec<Type>,
        ty: Option<Type>,
    },
}

impl FunctionSig {
    pub fn name(&self) -> &str {
        match self {
            Self::Extern(name) => name,
            Self::Decl { name, .. } => name,
        }
    }
    pub fn get(func: &PFunction) -> Self {
        Self::Decl {
            name: func.name.to_string(),
            arg_types: func.args.iter().map(|arg| arg.r#type()).collect(),
            ty: func.ret,
        }
    }
    pub(super) fn from_pimport(import: &PImport) -> Self {
        Self::Extern(import.name().to_string())
    }
}

#[derive(Debug, Clone)]
pub enum ExternArg {
    String(String),
    Array(String),
    Expr(HIRExpr),
}

impl From<PString<'_>> for ExternArg {
    fn from(value: PString) -> Self {
        Self::String(value.span().to_string())
    }
}

#[derive(Debug, Clone)]
pub struct HIRRoot {
    pub globals: VarSymMap,
    pub functions: FuncSymMap,
    pub imports: ImportSymMap,
}

#[derive(Debug, Clone)]
pub struct HIRAssign {
    pub lhs: HIRLoc,
    pub rhs: HIRExpr,
}
