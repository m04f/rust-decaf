use crate::span::*;

mod checker;

pub use checker::Error;

#[cfg(test)]
use proptest_derive::Arbitrary;

pub use self::checker::*;

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
#[cfg_attr(test, derive(Arbitrary))]
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

pub enum BlockElem<S> {
    Stmt(Stmt<S>),
    Func(Function<S>),
    Decl { decls: Vec<Var<S>>, span: S },
}

impl<S> BlockElem<S> {
    pub fn pos(&self) -> &S {
        match self {
            BlockElem::Stmt(stmt) => stmt.span(),
            BlockElem::Func(func) => func.span(),
            BlockElem::Decl { span, .. } => span,
        }
    }
}

impl<S> From<Stmt<S>> for BlockElem<S> {
    fn from(value: Stmt<S>) -> Self {
        BlockElem::Stmt(value)
    }
}

impl<S> From<Function<S>> for BlockElem<S> {
    fn from(value: Function<S>) -> Self {
        BlockElem::Func(value)
    }
}

impl<S> From<(Vec<Var<S>>, S)> for BlockElem<S> {
    fn from((decls, span): (Vec<Var<S>>, S)) -> Self {
        BlockElem::Decl { span, decls }
    }
}

/// a literal that can be used as an expression
#[derive(Debug, Clone)]
pub enum ELiteral<S> {
    Int(IntLiteral<S>),
    Char(CharLiteral<S>),
    Bool(BoolLiteral<S>),
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
{
    fn eq(&self, other: &ELiteral<U>) -> bool {
        match (self, other) {
            (Self::Int(lit), ELiteral::Int(other)) => lit == other,
            (Self::Char(lit), ELiteral::Char(other)) => lit == other,
            (Self::Bool(lit), ELiteral::Bool(other)) => lit == other,
            _ => false,
        }
    }
}

impl<S> ELiteral<S> {
    pub fn span(&self) -> &S {
        match self {
            Self::Int(lit) => lit.span(),
            Self::Char(lit) => lit.span(),
            Self::Bool(lit) => lit.span(),
        }
    }

    pub fn int(lit: IntLiteral<S>) -> Self {
        Self::Int(lit)
    }

    pub fn char(lit: CharLiteral<S>) -> Self {
        Self::Char(lit)
    }

    pub fn bool(lit: BoolLiteral<S>) -> Self {
        Self::Bool(lit)
    }
}

#[derive(Debug, Clone)]
pub enum Arg<S> {
    String(StringLiteral<S>),
    Expr(Expr<S>),
}

impl<S> Arg<S> {
    pub fn from_expr(expr: Expr<S>) -> Self {
        Arg::Expr(expr)
    }

    pub fn from_string(lit: StringLiteral<S>) -> Self {
        Arg::String(lit)
    }

    pub fn span(&self) -> &S {
        match self {
            Self::String(lit) => lit.span(),
            Self::Expr(expr) => expr.span(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Call<S> {
    name: Identifier<S>,
    args: Vec<Arg<S>>,
    span: S,
}

impl<S> Call<S> {
    pub fn new(name: Identifier<S>, args: Vec<Arg<S>>, span: S) -> Self {
        Self { name, args, span }
    }

    pub fn span(&self) -> &S {
        &self.span
    }
}

impl<S> From<Call<S>> for Expr<S> {
    fn from(value: Call<S>) -> Self {
        Expr::Call(value)
    }
}

#[derive(Debug, Clone)]
pub struct Loc<S> {
    ident: Identifier<S>,
    offset: Option<Expr<S>>,
    span: S,
}

impl<S> Loc<S> {
    pub fn span(&self) -> &S {
        &self.span
    }

    pub fn with_offset(ident: Identifier<S>, offset: Expr<S>, span: S) -> Self {
        Self {
            ident,
            offset: Some(offset),
            span,
        }
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

impl<S> From<Loc<S>> for Expr<S> {
    fn from(value: Loc<S>) -> Self {
        match value.offset {
            Some(offset) => Expr::index(value.ident, offset, value.span),
            None => Expr::Scalar(value.ident),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expr<S> {
    Len(S, Identifier<S>),
    Nested(S, Box<Expr<S>>),
    Not(S, Box<Expr<S>>),
    Neg(S, Box<Expr<S>>),
    Ter {
        cond: Box<Expr<S>>,
        yes: Box<Expr<S>>,
        no: Box<Expr<S>>,
        span: S,
    },
    Call(Call<S>),
    Index {
        name: Identifier<S>,
        offset: Box<Expr<S>>,
        span: S,
    },

    Scalar(Identifier<S>),
    Literal(ELiteral<S>),
    BinOp {
        op: Op,
        lhs: Box<Expr<S>>,
        rhs: Box<Expr<S>>,
        span: S,
    },
}

impl<S> From<IntLiteral<S>> for Expr<S> {
    fn from(lit: IntLiteral<S>) -> Self {
        Expr::Literal(ELiteral::Int(lit))
    }
}

impl<S> From<CharLiteral<S>> for Expr<S> {
    fn from(lit: CharLiteral<S>) -> Self {
        Expr::Literal(ELiteral::Char(lit))
    }
}

impl<S> From<BoolLiteral<S>> for Expr<S> {
    fn from(lit: BoolLiteral<S>) -> Self {
        Expr::Literal(ELiteral::Bool(lit))
    }
}

impl<S> From<Identifier<S>> for Expr<S> {
    fn from(value: Identifier<S>) -> Self {
        Expr::Scalar(value)
    }
}

impl<T, U> PartialEq<Expr<U>> for Expr<T>
where
    U: AsRef<[u8]>,
    T: AsRef<[u8]>,
{
    fn eq(&self, other: &Expr<U>) -> bool {
        use Expr::*;
        match (self, other) {
            (Len(_, l), Len(_, r)) => l.span().as_ref() == r.span().as_ref(),
            (Nested(_, l), Nested(_, r)) => l.as_ref() == r.as_ref(),
            (Nested(_, l), e) => l.as_ref() == e,
            (e, Nested(_, r)) => e == r.as_ref(),
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
            (Literal(l), Literal(r)) => l == r,
            (Call(..), Call(..)) => unimplemented!(),
            _ => false,
        }
    }
}

impl<S> Expr<S> {
    pub fn is_binop(&self) -> bool {
        matches!(self, Self::BinOp { .. })
    }
    pub fn span(&self) -> &S {
        match self {
            Self::Len(s, _) => s,
            Self::Nested(s, _) => s,
            Self::Not(s, _) => s,
            Self::Neg(s, _) => s,
            Self::Ter { span, .. } => span,
            Self::BinOp { span, .. } => span,
            Self::Scalar(i) => i.span(),
            Self::Literal(l) => l.span(),
            Self::Index { span, .. } => span,
            Self::Call(call) => call.span(),
        }
    }
    pub fn ident(name: Identifier<S>) -> Self {
        Self::Scalar(name)
    }

    pub fn nested(expr: Self, span: S) -> Self {
        Self::Nested(span, Box::new(expr))
    }

    pub fn len(name: Identifier<S>, span: S) -> Self {
        Self::Len(span, name)
    }

    pub fn loc(name: Identifier<S>, index: Option<Self>, span: S) -> Self {
        match index {
            Some(index) => Self::index(name, index, span),
            None => Self::Scalar(name),
        }
    }

    pub fn index(name: Identifier<S>, index: Self, span: S) -> Self {
        Self::Index {
            name,
            offset: Box::new(index),
            span,
        }
    }

    pub fn call(name: Identifier<S>, args: Vec<Arg<S>>, span: S) -> Self {
        Self::Call(Call::new(name, args, span))
    }

    pub fn ter(cond: Self, yes: Self, no: Self, span: S) -> Self {
        Self::Ter {
            cond: Box::new(cond),
            yes: Box::new(yes),
            no: Box::new(no),
            span,
        }
    }
    pub fn binop(lhs: Self, op: Op, rhs: Self, span: S) -> Self {
        Self::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            span,
        }
    }
    pub fn neg(expr: Self, span: S) -> Self {
        Self::Neg(span, Box::new(expr))
    }
    pub fn not(expr: Self, span: S) -> Self {
        Self::Not(span, Box::new(expr))
    }
    pub fn literal(literal: ELiteral<S>) -> Self {
        Self::Literal(literal)
    }

    // returns the precedence of the current operation in the expr
    // higher precedence means that the operation should be evaluated first.
    pub fn precedence(&self) -> usize {
        use Expr::*;
        match self {
            BinOp { op, .. } => op.precedence(),
            Ter { .. } => 1,
            Neg(..)
            | Not(..)
            | Len(..)
            | Nested(..)
            | Call { .. }
            | Literal(..)
            | Scalar(..)
            | Index { .. } => usize::MAX,
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[cfg_attr(test, derive(Arbitrary))]
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

#[derive(Debug, Clone)]
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
}

#[derive(Debug, Clone)]
pub struct Function<S> {
    name: Identifier<S>,
    args: Vec<Var<S>>,
    body: Block<S>,
    ret: Option<Type>,
    span: S,
}

impl<S> Function<S> {
    pub fn span(&self) -> &S {
        &self.span
    }

    pub fn new(
        name: Identifier<S>,
        args: Vec<Var<S>>,
        body: Block<S>,
        ret: Option<Type>,
        span: S,
    ) -> Self {
        Self {
            name,
            args,
            body,
            ret,
            span,
        }
    }
}

#[derive(Debug, Clone)]
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

#[derive(Debug, Clone, Default)]
pub struct Block<S> {
    funcs: Vec<Function<S>>,
    decls: Vec<Var<S>>,
    stmts: Vec<Stmt<S>>,
}

impl<S: Default> Block<S> {
    pub fn new() -> Self {
        Block::default()
    }
    pub fn add(&mut self, elem: BlockElem<S>) {
        match elem {
            BlockElem::Func(func) => self.funcs.push(func),
            BlockElem::Decl { decls, span } => self.decls.extend(decls),
            BlockElem::Stmt(stmt) => self.stmts.push(stmt),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AssignExpr<S> {
    Inc,
    Dec,
    AddAssign(Expr<S>),
    SubAssign(Expr<S>),
    Assign(Expr<S>),
}

impl<S> AssignExpr<S> {
    pub fn inc() -> Self {
        Self::Inc
    }
    pub fn dec() -> Self {
        Self::Dec
    }
    pub fn add_assign(expr: Expr<S>) -> Self {
        Self::AddAssign(expr)
    }
    pub fn sub_assign(expr: Expr<S>) -> Self {
        Self::SubAssign(expr)
    }
    pub fn assign(expr: Expr<S>) -> Self {
        Self::Assign(expr)
    }
}

#[derive(Debug, Clone)]
pub struct Assign<S> {
    lhs: Loc<S>,
    op: AssignExpr<S>,
    span: S,
}

impl<S> Assign<S> {
    pub fn span(&self) -> &S {
        &self.span
    }

    pub fn new(lhs: Loc<S>, op: AssignExpr<S>, span: S) -> Self {
        Self { lhs, op, span }
    }
}

impl<S> From<Assign<S>> for Stmt<S> {
    fn from(assign: Assign<S>) -> Self {
        Self::Assign(assign)
    }
}

#[derive(Debug, Clone)]
pub enum Stmt<S> {
    Call(Call<S>),
    If {
        cond: Expr<S>,
        yes: Block<S>,
        no: Option<Block<S>>,
        span: S,
    },
    While {
        cond: Expr<S>,
        body: Block<S>,
        span: S,
    },
    For {
        init: Assign<S>,
        cond: Expr<S>,
        update: Assign<S>,
        body: Block<S>,
        span: S,
    },
    Assign(Assign<S>),
    Return {
        expr: Option<Expr<S>>,
        span: S,
    },
    Break(S),
    Continue(S),
}

impl<S> Stmt<S> {
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
    pub fn call_stmt(call: Call<S>) -> Self {
        Self::Call(call)
    }

    pub fn if_stmt(cond: Expr<S>, yes: Block<S>, no: Option<Block<S>>, span: S) -> Self {
        Self::If {
            cond,
            yes,
            no,
            span,
        }
    }
    pub fn while_stmt(cond: Expr<S>, body: Block<S>, span: S) -> Self {
        Self::While { cond, body, span }
    }
    pub fn return_stmt(expr: Option<Expr<S>>, span: S) -> Self {
        Self::Return { expr, span }
    }
    pub fn continue_stmt(span: S) -> Self {
        Self::Continue(span)
    }
    pub fn break_stmt(span: S) -> Self {
        Self::Break(span)
    }
    pub fn for_stmt(
        init: Assign<S>,
        cond: Expr<S>,
        update: Assign<S>,
        body: Block<S>,
        span: S,
    ) -> Self {
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
pub enum DocElem<S> {
    Function(Function<S>),
    Decl(Vec<Var<S>>, S),
    Import(Import<S>),
}

impl<S> DocElem<S> {
    pub fn function(func: Function<S>) -> Self {
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
pub struct Doc<S> {
    elems: Vec<DocElem<S>>,
}
