use std::collections::{HashMap, HashSet};

use crate::span::Span;
use crate::cst;

pub type SymMap<T> = HashMap<String, T>;
pub type VarSymMap = SymMap<Var>;
pub type FuncSymMap = SymMap<Function>;
pub type ImportSymMap = HashSet<String>;
pub type SigSymMap = SymMap<FunctionSig>;

pub use crate::cst::Type;

pub type Identifier = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Literal {
    Int(i64),
    Bool(bool),
}

impl From<i64> for Literal {
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
pub enum Var {
    Scalar(Typed<Identifier>),
    Array { arr: Typed<Identifier>, size: u64 },
}

impl Var {
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
pub enum Location {
    Scalar(Typed<Identifier>),
    Index {
        arr: Typed<Identifier>,
        size: u64,
        index: Expr,
    },
}

impl Location {
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
    pub fn index(var: Var, index: Expr) -> Self {
        assert!(var.is_array());
        match var {
            Var::Scalar(_) => unreachable!(),
            Var::Array { arr, size } => Self::Index {
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

impl Literal {
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

impl TryFrom<cst::Op> for ArithOp {
    type Error = ();
    fn try_from(op: cst::Op) -> Result<Self, Self::Error> {
        match op {
            cst::Op::Add => Ok(ArithOp::Add),
            cst::Op::Sub => Ok(ArithOp::Sub),
            cst::Op::Mul => Ok(ArithOp::Mul),
            cst::Op::Div => Ok(ArithOp::Div),
            cst::Op::Mod => Ok(ArithOp::Mod),
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

impl TryFrom<cst::Op> for RelOp {
    type Error = ();
    fn try_from(op: cst::Op) -> Result<Self, Self::Error> {
        match op {
            cst::Op::Less => Ok(RelOp::Less),
            cst::Op::LessEqual => Ok(RelOp::LessEqual),
            cst::Op::Greater => Ok(RelOp::Greater),
            cst::Op::GreaterEqual => Ok(RelOp::GreaterEqual),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum EqOp {
    Equal,
    NotEqual,
}

impl TryFrom<cst::Op> for EqOp {
    type Error = ();
    fn try_from(op: cst::Op) -> Result<Self, Self::Error> {
        match op {
            cst::Op::Equal => Ok(EqOp::Equal),
            cst::Op::NotEqual => Ok(EqOp::NotEqual),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum CondOp {
    And,
    Or,
}

impl TryFrom<cst::Op> for CondOp {
    type Error = ();
    fn try_from(op: cst::Op) -> Result<Self, Self::Error> {
        match op {
            cst::Op::And => Ok(CondOp::And),
            cst::Op::Or => Ok(CondOp::Or),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr {
    Len(u64),
    Not(Box<Expr>),
    Neg(Box<Expr>),
    Ter {
        cond: Box<Expr>,
        yes: Box<Expr>,
        no: Box<Expr>,
    },
    Call(Call),
    Loc(Box<Location>),
    IntLiteral(i64),
    BoolLiteral(bool),
    Arith {
        op: ArithOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Rel {
        op: RelOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Eq {
        op: EqOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },

    Cond {
        op: CondOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

impl From<Location> for Expr {
    fn from(value: Location) -> Self {
        Self::Loc(Box::new(value))
    }
}

impl From<Literal> for Expr {
    fn from(value: Literal) -> Self {
        match value {
            Literal::Int(n) => Expr::IntLiteral(n),
            Literal::Bool(b) => Expr::BoolLiteral(b),
        }
    }
}

impl Expr {
    pub fn new_not(self) -> Self {
        assert!(self.r#type() == Type::Bool);
        Self::Not(Box::new(self))
    }

    pub fn new_neg(self) -> Self {
        assert!(self.r#type() == Type::Int);
        Self::Neg(Box::new(self))
    }

    pub fn is_boolean(&self) -> bool {
        self.r#type() == Type::Bool
    }

    pub fn is_int(&self) -> bool {
        self.r#type() == Type::Int
    }

    pub fn r#type(&self) -> Type {
        use Expr::*;
        match self {
            Len(_) | Neg(_) | Arith { .. } | IntLiteral(_) => Type::Int,
            Cond { .. } | Eq { .. } | Rel { .. } | Not(..) | BoolLiteral(_) => Type::Bool,
            Ter { yes, .. } => yes.r#type(),
            Loc(loc) => loc.r#type(),
            Call(call) => call.return_type().unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Call {
    Extern {
        name: String,
        args: Vec<ExternArg>,
    },
    Decaf {
        name: String,
        ret: Option<Type>,
        args: Vec<Expr>,
    },
}

impl Call {
    fn return_type(&self) -> Option<Type> {
        match self {
            Self::Extern { .. } => Some(Type::Int),
            Self::Decaf { ret, .. } => *ret,
        }
    }
    pub fn new_extern(name: String, args: Vec<ExternArg>) -> Self {
        Self::Extern { name, args }
    }
    pub fn new_decaf(name: String, ret: Option<Type>, args: Vec<Expr>) -> Self {
        Self::Decaf { name, ret, args }
    }
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Assign(Assign),
    Expr(Expr),
    Return(Option<Expr>),
    Break,
    Continue,
    If {
        cond: Expr,
        yes: Box<Block>,
        no: Box<Block>,
    },
    While {
        cond: Expr,
        body: Box<Block>,
    },
    For {
        init: Assign,
        cond: Expr,
        update: Assign,
        body: Box<Block>,
    },
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub body: Block,
    pub args: VarSymMap,
    pub ret: Option<Type>,
}

impl Function {
    pub fn new(name: Span, body: Block, args: VarSymMap, ret: Option<Type>) -> Self {
        Self {
            name: name.to_string(),
            body,
            args,
            ret,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Block {
    pub decls: VarSymMap,
    pub stmts: Vec<Stmt>,
}

impl Block {
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
    pub fn get(func: &cst::PFunction) -> Self {
        Self::Decl {
            name: func.name.to_string(),
            arg_types: func.args.iter().map(|arg| arg.r#type()).collect(),
            ty: func.ret,
        }
    }
    pub(super) fn from_pimport(import: &cst::Import) -> Self {
        Self::Extern(import.name().to_string())
    }
}

#[derive(Debug, Clone)]
pub enum ExternArg {
    String(String),
    Array(String),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Root {
    pub globals: VarSymMap,
    pub functions: FuncSymMap,
    pub imports: ImportSymMap,
}

#[derive(Debug, Clone)]
pub struct Assign {
    pub lhs: Location,
    pub rhs: Expr,
}
