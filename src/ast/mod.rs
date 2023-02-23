use crate::span::*;

mod checker;

pub use checker::Error;

pub use self::checker::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    And,
    Or,
}

impl Op {
    fn precedence(self) -> usize {
        use Op::*;
        match self {
            Mul | Div | Mod => 10,
            Add | Sub => 9,
            Less | LessEqual | Greater | GreaterEqual => 8,
            Equal | NotEqual => 7,
            And => 6,
            Or => 5,
        }
    }
}

pub enum BlockElem<S, Block, Args, Stmt> {
    Stmt(Stmt),
    Func(Function<Block, Args, S>),
    Decl { decls: Vec<Var<S>>, span: S },
}

/// a literal that can be used as an expression
#[derive(Debug, Clone, Copy)]
pub enum ELiteral<S> {
    Decimal(S),
    Hex(S),
    Char(u8),
    Bool(bool),
}

impl<T, U> PartialEq<IntLiteral<U>> for IntLiteral<T>
where
    T: AsRef<[u8]>,
    U: AsRef<[u8]>,
{
    fn eq(&self, other: &IntLiteral<U>) -> bool {
        match (self, other) {
            (Self::Decimal(lit), IntLiteral::Decimal(other)) => lit.as_ref() == other.as_ref(),
            (Self::Hex(lit), IntLiteral::Hex(other)) => lit.as_ref() == other.as_ref(),
            _ => false,
        }
    }
}

impl<T, U> PartialEq<CharLiteral<U>> for CharLiteral<T> {
    fn eq(&self, other: &CharLiteral<U>) -> bool {
        self.1 == other.1
    }
}

impl<T, U> PartialEq<BoolLiteral<U>> for BoolLiteral<T> {
    fn eq(&self, other: &BoolLiteral<U>) -> bool {
        self.1 == other.1
    }
}

impl<T, U> PartialEq<ELiteral<U>> for ELiteral<T>
where
    IntLiteral<T>: PartialEq<IntLiteral<U>>,
    CharLiteral<T>: PartialEq<CharLiteral<U>>,
    BoolLiteral<T>: PartialEq<BoolLiteral<U>>,
    T: PartialEq<U>,
{
    fn eq(&self, other: &ELiteral<U>) -> bool {
        match (self, other) {
            (Self::Decimal(lit), ELiteral::Decimal(other)) => *lit == *other,
            (Self::Hex(lit), ELiteral::Hex(other)) => *lit == *other,
            (Self::Char(lit), ELiteral::Char(other)) => lit == other,
            (Self::Bool(lit), ELiteral::Bool(other)) => lit == other,
            _ => false,
        }
    }
}

impl<S> ELiteral<S> {
    pub fn decimal(lit: S) -> Self {
        Self::Decimal(lit)
    }
    pub fn hex(lit: S) -> Self {
        Self::Hex(lit)
    }

    pub fn char(lit: u8) -> Self {
        Self::Char(lit)
    }

    pub fn bool(lit: bool) -> Self {
        Self::Bool(lit)
    }
}

#[derive(Debug, Clone)]
pub struct Call<S, Arg> {
    pub name: Identifier<S>,
    pub args: Vec<Arg>,
    pub span: S,
}

impl<S, Arg> Call<S, Arg> {
    pub fn new(name: Identifier<S>, args: Vec<Arg>, span: S) -> Self {
        Self { name, args, span }
    }

    pub fn span(&self) -> &S {
        &self.span
    }
}

#[derive(Debug, Clone)]
pub struct Loc<Lit, Arg, S, Ext> {
    pub ident: Identifier<S>,
    pub offset: Option<Expr<Lit, Arg, S, Ext>>,
    span: S,
}

impl<Lit, Arg, S, Ext> Loc<Lit, Arg, S, Ext> {
    pub fn span(&self) -> &S {
        &self.span
    }

    pub fn with_offset(ident: Identifier<S>, offset: Expr<Lit, Arg, S, Ext>, span: S) -> Self {
        Self {
            ident,
            offset: Some(offset),
            span,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.offset.is_none()
    }

    pub fn is_indexed(&self) -> bool {
        self.offset.is_some()
    }

    pub fn from_ident(ident: Identifier<S>) -> Self
    where
        S: Clone,
    {
        Self {
            ident: ident.clone(),
            offset: None,
            span: ident.span().clone(),
        }
    }
}

impl<Lit, Arg, S> From<Loc<Lit, Arg, S, ()>> for Expr<Lit, Arg, S, ()> {
    fn from(value: Loc<Lit, Arg, S, ()>) -> Self {
        match value.offset {
            Some(offset) => Self::index(value.ident, offset, value.span),
            None => ExprInner::Scalar(value.ident).into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Expr<Lit, Arg, S, Ext> {
    pub extra: Ext,
    pub inner: ExprInner<Lit, Arg, S, Ext>,
}

#[derive(Debug, Clone)]
pub enum ExprInner<Lit, Arg, S, Ext> {
    Len(S, Identifier<S>),
    Nested(S, Box<Expr<Lit, Arg, S, Ext>>),
    Not(S, Box<Expr<Lit, Arg, S, Ext>>),
    Neg(S, Box<Expr<Lit, Arg, S, Ext>>),
    Ter {
        cond: Box<Expr<Lit, Arg, S, Ext>>,
        yes: Box<Expr<Lit, Arg, S, Ext>>,
        no: Box<Expr<Lit, Arg, S, Ext>>,
        span: S,
    },
    Call(Call<S, Arg>),
    Index {
        name: Identifier<S>,
        offset: Box<Expr<Lit, Arg, S, Ext>>,
        span: S,
    },

    Scalar(Identifier<S>),
    Literal {
        span: S,
        value: Lit,
    },
    BinOp {
        op: Op,
        lhs: Box<Expr<Lit, Arg, S, Ext>>,
        rhs: Box<Expr<Lit, Arg, S, Ext>>,
        span: S,
    },
}

impl<Lit, Arg, S, Ext> ExprInner<Lit, Arg, S, Ext> {
    pub fn span(&self) -> &S {
        match self {
            Self::Len(s, _)
            | Self::Nested(s, _)
            | Self::Neg(s, _)
            | Self::Not(s, _)
            | Self::Ter { span: s, .. }
            | Self::BinOp { span: s, .. }
            | Self::Index { span: s, .. }
            | Self::Literal { span: s, .. } => s,
            Self::Scalar(ident) => ident.span(),
            Self::Call(call) => call.span(),
        }
    }

    pub fn scalar(&self) -> Option<&Identifier<S>> {
        match self {
            Self::Scalar(ident) => Some(ident),
            _ => None,
        }
    }
}

impl<Lit, Arg, S> From<ExprInner<Lit, Arg, S, ()>> for Expr<Lit, Arg, S, ()> {
    fn from(value: ExprInner<Lit, Arg, S, ()>) -> Self {
        Self {
            extra: (),
            inner: value,
        }
    }
}

impl<Lit, Arg, S> From<Identifier<S>> for Expr<Lit, Arg, S, ()> {
    fn from(value: Identifier<S>) -> Self {
        ExprInner::Scalar(value).into()
    }
}

impl<Lit, Arg, T, U> PartialEq<Expr<Lit, Arg, U, ()>> for Expr<Lit, Arg, T, ()>
where
    U: AsRef<[u8]>,
    T: AsRef<[u8]>,
    Lit: PartialEq,
{
    fn eq(&self, other: &Expr<Lit, Arg, U, ()>) -> bool {
        use ExprInner::*;
        match (&self.inner, &other.inner) {
            (Len(_, l), Len(_, r)) => l.span().as_ref() == r.span().as_ref(),
            (Nested(_, l), Nested(_, r)) => l.as_ref() == r.as_ref(),
            (Nested(_, l), _) => l.as_ref() == other,
            (_, Nested(_, r)) => self == r.as_ref(),
            (Not(_, l), Not(_, r)) => l.as_ref() == r.as_ref(),
            (Neg(_, l), Neg(_, r)) => l.as_ref() == r.as_ref(),
            (
                Ter {
                    cond: lcond,
                    yes: lyes,
                    no: lno,
                    ..
                },
                Ter {
                    cond: rcond,
                    yes: ryes,
                    no: rno,
                    ..
                },
            ) => {
                lcond.as_ref() == rcond.as_ref()
                    && lyes.as_ref() == ryes.as_ref()
                    && lno.as_ref() == rno.as_ref()
            }
            (
                BinOp {
                    op: lop,
                    lhs: llhs,
                    rhs: lrhs,
                    ..
                },
                BinOp {
                    op: rop,
                    lhs: rlhs,
                    rhs: rrhs,
                    ..
                },
            ) => lop == rop && llhs.as_ref() == rlhs.as_ref() && lrhs.as_ref() == rrhs.as_ref(),
            (Scalar(l), Scalar(r)) => l.span().as_ref() == r.span().as_ref(),
            (
                Index {
                    name: l,
                    offset: li,
                    ..
                },
                Index {
                    name: r,
                    offset: ri,
                    ..
                },
            ) => l.span().as_ref() == r.span().as_ref() && li.as_ref() == ri.as_ref(),
            (Literal { value: lvalue, .. }, Literal { value: rvalue, .. }) => *lvalue == *rvalue,
            (Call(..), Call(..)) => unimplemented!(),
            _ => false,
        }
    }
}

impl<Lit, Arg, S, Ext> Expr<Lit, Arg, S, Ext> {
    pub fn span(&self) -> &S {
        match &self.inner {
            ExprInner::Len(s, _) => s,
            ExprInner::Nested(s, _) => s,
            ExprInner::Not(s, _) => s,
            ExprInner::Neg(s, _) => s,
            ExprInner::Ter { span, .. } => span,
            ExprInner::BinOp { span, .. } => span,
            ExprInner::Scalar(i) => i.span(),
            ExprInner::Literal { span, .. } => span,
            ExprInner::Index { span, .. } => span,
            ExprInner::Call(call) => call.span(),
        }
    }
}

impl<Lit, Arg, S> Expr<Lit, Arg, S, ()> {
    pub fn is_binop(&self) -> bool {
        matches!(self.inner, ExprInner::BinOp { .. })
    }
    pub fn ident(name: Identifier<S>) -> Self {
        ExprInner::Scalar(name).into()
    }

    pub fn nested(expr: Self, span: S) -> Self {
        ExprInner::Nested(span, Box::new(expr)).into()
    }

    pub fn len(name: Identifier<S>, span: S) -> Self {
        ExprInner::Len(span, name).into()
    }

    pub fn loc(name: Identifier<S>, index: Option<Self>, span: S) -> Self {
        match index {
            Some(index) => Self::index(name, index, span),
            None => ExprInner::Scalar(name).into(),
        }
    }

    pub fn index(name: Identifier<S>, index: Self, span: S) -> Self {
        ExprInner::Index {
            name,
            offset: Box::new(index),
            span,
        }
        .into()
    }

    pub fn ter(cond: Self, yes: Self, no: Self, span: S) -> Self {
        ExprInner::Ter {
            cond: Box::new(cond),
            yes: Box::new(yes),
            no: Box::new(no),
            span,
        }
        .into()
    }
    pub fn binop(lhs: Self, op: Op, rhs: Self, span: S) -> Self {
        ExprInner::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            span,
        }
        .into()
    }
    pub fn neg(expr: Self, span: S) -> Self {
        ExprInner::Neg(span, Box::new(expr)).into()
    }
    pub fn not(expr: Self, span: S) -> Self {
        ExprInner::Not(span, Box::new(expr)).into()
    }

    // returns the precedence of the current operation in the expr
    // higher precedence means that the operation should be evaluated first.
    pub fn precedence(&self) -> usize {
        use ExprInner::*;
        match self.inner {
            BinOp { op, .. } => op.precedence(),
            Ter { .. } => 1,
            Neg(..)
            | Not(..)
            | Len(..)
            | Nested(..)
            | Call { .. }
            | Literal { .. }
            | Scalar(..)
            | Index { .. } => usize::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Bool,
    Int,
}

impl Type {
    pub fn int_type() -> Self {
        Self::Int
    }
    pub fn bool_type() -> Self {
        Self::Bool
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Identifier<S>(S);

impl<T, U> PartialEq<Identifier<U>> for Identifier<T>
where
    T: AsRef<[u8]>,
    U: AsRef<[u8]>,
{
    fn eq(&self, other: &Identifier<U>) -> bool {
        self.0.as_ref() == other.0.as_ref()
    }
}

impl<S> From<S> for Identifier<S>
where
    S: AsRef<[u8]>,
{
    fn from(s: S) -> Self {
        Self(s)
    }
}

impl<S> Identifier<S> {
    pub fn from_span(span: S) -> Self {
        Self(span)
    }
    pub fn span(&self) -> &S {
        &self.0
    }
}
impl<'a> Identifier<Span<'a>> {
    pub fn fragment(&self) -> &'a [u8] {
        self.0.source()
    }
}

#[derive(Debug, Clone)]
pub struct Import<S>(S, Identifier<S>);

impl<'a> Import<Span<'a>> {
    pub fn from_ident(ident: Spanned<'a, Identifier<Span<'a>>>) -> Self {
        let (ident, span) = ident.into_parts();
        Self(span, ident)
    }
}
impl<S> Import<S> {
    pub fn span(&self) -> &S {
        &self.0
    }
    pub fn name(&self) -> &Identifier<S> {
        &self.1
    }
}

#[derive(Debug, Clone)]
pub enum Var<S> {
    Array {
        ty: Type,
        ident: Identifier<S>,
        size: IntLiteral<S>,
        // we do not need to record spans for identifiers
        span: S,
    },
    Scalar {
        ty: Type,
        ident: Identifier<S>,
    },
}

impl<T, U> PartialEq<Var<U>> for Var<T>
where
    T: AsRef<[u8]>,
    U: AsRef<[u8]>,
{
    fn eq(&self, other: &Var<U>) -> bool {
        match (self, other) {
            (
                Self::Array { ident, size, .. },
                Var::Array {
                    ident: ident2,
                    size: size2,
                    ..
                },
            ) => ident == ident2 && size == size2,
            (Self::Scalar { ident, .. }, Var::Scalar { ident: ident2, .. }) => ident == ident2,
            _ => false,
        }
    }
}

impl<S> Var<S> {
    pub fn new(ty: Type, ident: Identifier<S>, size: Option<IntLiteral<S>>, span: S) -> Self {
        match size {
            Some(size) => Self::array(ty, ident, size, span),
            None => Self::scalar(ty, ident),
        }
    }
    pub fn array(ty: Type, ident: Identifier<S>, size: IntLiteral<S>, span: S) -> Self {
        Self::Array {
            ty,
            ident,
            size,
            span,
        }
    }
    pub fn scalar(ty: Type, ident: Identifier<S>) -> Self {
        Self::Scalar { ty, ident }
    }
    pub fn span(&self) -> &S {
        match self {
            Self::Array { span, .. } => span,
            Self::Scalar { ident, .. } => ident.span(),
        }
    }
    pub fn name(&self) -> &S {
        match self {
            Self::Array { ident, .. } => ident.span(),
            Self::Scalar { ident, .. } => ident.span(),
        }
    }
    pub fn ty(&self) -> Type {
        match self {
            Self::Array { ty, .. } => *ty,
            Self::Scalar { ty, .. } => *ty,
        }
    }
    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Scalar { .. })
    }
    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array { .. })
    }
}

#[derive(Debug, Clone)]
pub struct Function<Block, Args, S> {
    pub name: Identifier<S>,
    pub body: Block,
    pub args: Args,
    pub ret: Option<Type>,
    span: S,
}

impl<Block, Args, S> Function<Block, Args, S> {
    pub fn span(&self) -> &S {
        &self.span
    }

    pub fn new(name: Identifier<S>, body: Block, args: Args, ret: Option<Type>, span: S) -> Self {
        Self {
            name,
            body,
            args,
            ret,
            span,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum IntLiteral<S> {
    Decimal(S),
    Hex(S),
}

impl<S> IntLiteral<S> {
    pub fn span(&self) -> &S {
        match self {
            Self::Decimal(s) => s,
            Self::Hex(s) => s,
        }
    }
    pub fn from_decimal(literal: S) -> Self {
        Self::Decimal(literal)
    }
    pub fn from_hex(literal: S) -> Self {
        Self::Hex(literal)
    }
}

impl<S> From<IntLiteral<S>> for ELiteral<S> {
    fn from(value: IntLiteral<S>) -> Self {
        match value {
            IntLiteral::Decimal(s) => ELiteral::Decimal(s),
            IntLiteral::Hex(s) => ELiteral::Hex(s),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CharLiteral<S>(S, u8);

impl<'a> CharLiteral<Span<'a>> {
    pub fn from_spanned(literal: Spanned<'a, u8>) -> Self {
        let (ch, span) = literal.into_parts();
        Self(span, ch)
    }
}
impl<S> CharLiteral<S> {
    pub fn span(&self) -> &S {
        &self.0
    }
    pub fn value(&self) -> u8 {
        self.1
    }
}

#[derive(Debug, Clone)]
pub struct BoolLiteral<S>(S, bool);

impl<'a> BoolLiteral<Span<'a>> {
    pub fn from_spanned(literal: Spanned<'a, bool>) -> Self {
        let (literal, span) = literal.into_parts();
        Self(span, literal)
    }
}

impl<S> BoolLiteral<S> {
    pub fn span(&self) -> &S {
        &self.0
    }
    pub fn value(&self) -> bool {
        self.1
    }
}

#[derive(Debug, Clone)]
pub struct StringLiteral<S>(S);

impl<S> StringLiteral<S> {
    pub fn from_span(span: S) -> Self {
        Self(span)
    }
    pub fn span(&self) -> &S {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct Block<Lit, Arg, S, Decls, Ext> {
    pub decls: Decls,
    pub stmts: Vec<Stmt<Lit, Arg, S, Decls, Ext>>,
}

impl<Lit, Arg, S, Ext, Decls: Default> Block<Lit, Arg, S, Decls, Ext> {
    pub fn new() -> Self {
        Self {
            decls: Decls::default(),
            stmts: Vec::new(),
        }
    }
}

impl<Lit, Arg, S, Ext, Decls: Default> Default for Block<Lit, Arg, S, Decls, Ext> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum AssignExpr<Lit, Arg, S, Ext> {
    Inc,
    Dec,
    AddAssign(Expr<Lit, Arg, S, Ext>),
    SubAssign(Expr<Lit, Arg, S, Ext>),
    Assign(Expr<Lit, Arg, S, Ext>),
}

impl<Lit, Arg, S, Ext> AssignExpr<Lit, Arg, S, Ext> {
    pub fn inc() -> Self {
        Self::Inc
    }
    pub fn dec() -> Self {
        Self::Dec
    }
    pub fn add_assign(expr: Expr<Lit, Arg, S, Ext>) -> Self {
        Self::AddAssign(expr)
    }
    pub fn sub_assign(expr: Expr<Lit, Arg, S, Ext>) -> Self {
        Self::SubAssign(expr)
    }
    pub fn assign(expr: Expr<Lit, Arg, S, Ext>) -> Self {
        Self::Assign(expr)
    }
}

#[derive(Debug, Clone)]
pub struct Assign<Lit, Arg, S, Ext> {
    pub lhs: Loc<Lit, Arg, S, Ext>,
    pub op: AssignExpr<Lit, Arg, S, Ext>,
    pub span: S,
}

impl<Lit, Arg, S, Ext> Assign<Lit, Arg, S, Ext> {
    pub fn span(&self) -> &S {
        &self.span
    }

    pub fn new(lhs: Loc<Lit, Arg, S, Ext>, op: AssignExpr<Lit, Arg, S, Ext>, span: S) -> Self {
        Self { lhs, op, span }
    }
}

impl<Lit, Arg, S, Decls, Ext> From<Assign<Lit, Arg, S, Ext>> for Stmt<Lit, Arg, S, Decls, Ext> {
    fn from(assign: Assign<Lit, Arg, S, Ext>) -> Self {
        Self::Assign(assign)
    }
}

#[derive(Debug, Clone)]
pub enum Stmt<Lit, Arg, S, Decls, Ext> {
    Call(Call<S, Arg>),
    If {
        cond: Expr<Lit, Arg, S, Ext>,
        yes: Block<Lit, Arg, S, Decls, Ext>,
        no: Option<Block<Lit, Arg, S, Decls, Ext>>,
        span: S,
    },
    While {
        cond: Expr<Lit, Arg, S, Ext>,
        body: Block<Lit, Arg, S, Decls, Ext>,
        span: S,
    },
    For {
        init: Assign<Lit, Arg, S, Ext>,
        cond: Expr<Lit, Arg, S, Ext>,
        update: Assign<Lit, Arg, S, Ext>,
        body: Block<Lit, Arg, S, Decls, Ext>,
        span: S,
    },
    Assign(Assign<Lit, Arg, S, Ext>),
    Return {
        expr: Option<Expr<Lit, Arg, S, Ext>>,
        span: S,
    },
    Break(S),
    Continue(S),
}

impl<Lit, Arg, S, Decls, Ext> Stmt<Lit, Arg, S, Decls, Ext> {
    pub fn span(&self) -> &S {
        match self {
            Self::Call(call) => call.span(),
            Self::If { span, .. } => span,
            Self::While { span, .. } => span,
            Self::For { span, .. } => span,
            Self::Assign(assign) => assign.span(),
            Self::Return { span, .. } => span,
            Self::Break(span) => span,
            Self::Continue(span) => span,
        }
    }
    pub fn call_stmt(call: Call<S, Arg>) -> Self {
        Self::Call(call)
    }

    pub fn if_stmt(
        cond: Expr<Lit, Arg, S, Ext>,
        yes: Block<Lit, Arg, S, Decls, Ext>,
        no: Option<Block<Lit, Arg, S, Decls, Ext>>,
        span: S,
    ) -> Self {
        Self::If {
            cond,
            yes,
            no,
            span,
        }
    }
    pub fn while_stmt(
        cond: Expr<Lit, Arg, S, Ext>,
        body: Block<Lit, Arg, S, Decls, Ext>,
        span: S,
    ) -> Self {
        Self::While { cond, body, span }
    }
    pub fn return_stmt(expr: Option<Expr<Lit, Arg, S, Ext>>, span: S) -> Self {
        Self::Return { expr, span }
    }
    pub fn continue_stmt(span: S) -> Self {
        Self::Continue(span)
    }
    pub fn break_stmt(span: S) -> Self {
        Self::Break(span)
    }
    pub fn for_stmt(
        init: Assign<Lit, Arg, S, Ext>,
        cond: Expr<Lit, Arg, S, Ext>,
        update: Assign<Lit, Arg, S, Ext>,
        body: Block<Lit, Arg, S, Decls, Ext>,
        span: S,
    ) -> Stmt<Lit, Arg, S, Decls, Ext> {
        Self::For {
            init,
            cond,
            update,
            body,
            span,
        }
    }
}

#[derive(Debug, Clone)]
pub enum DocElem<Block, Args, S> {
    Function(Function<Block, Args, S>),
    Decl(Vec<Var<S>>, S),
    Import(Import<S>),
}

impl<Block, Args, S> DocElem<Block, Args, S> {
    pub fn function(func: Function<Block, Args, S>) -> Self {
        Self::Function(func)
    }
    pub fn decl(decls: Vec<Var<S>>, span: S) -> Self {
        Self::Decl(decls, span)
    }
    pub fn import(import: Import<S>) -> Self {
        Self::Import(import)
    }

    pub fn span(&self) -> &S {
        match self {
            Self::Function(func) => func.span(),
            Self::Decl(_, span) => span,
            Self::Import(import) => import.span(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Root<S, Decls, Funcs> {
    pub imports: Vec<Import<S>>,
    pub decls: Decls,
    pub funcs: Funcs,
}

impl<S, Decls: Default, Funcs: Default> Default for Root<S, Decls, Funcs> {
    fn default() -> Self {
        Self {
            imports: Vec::new(),
            decls: Decls::default(),
            funcs: Funcs::default(),
        }
    }
}

impl<S, Decls, Funcs> Root<S, Decls, Funcs> {
    pub fn into_parts(self) -> (Vec<Import<S>>, Funcs, Decls) {
        (self.imports, self.funcs, self.decls)
    }
    pub fn imports(&self) -> &[Import<S>] {
        &self.imports
    }
    pub fn decls(&self) -> &Decls {
        &self.decls
    }
    pub fn funcs(&self) -> &Funcs {
        &self.funcs
    }
}
