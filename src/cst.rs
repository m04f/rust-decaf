use std::fmt::Display;

use crate::span::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Bool,
}

impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Bool => write!(f, "bool"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Import<'a> {
    span: Span<'a>,
    id: Span<'a>,
}

impl<'a> Import<'a> {
    pub fn name(&self) -> Span<'a> {
        self.id
    }
    pub fn span(&self) -> Span<'a> {
        self.span
    }
}

impl<'a> From<Spanned<'a, Span<'a>>> for Import<'a> {
    fn from(value: Spanned<'a, Span<'a>>) -> Self {
        let (ident, span) = value.into_parts();
        Self { span, id: ident }
    }
}

#[derive(Debug, Clone)]
pub enum AssignExpr<'a> {
    Inc,
    Dec,
    AddAssign(Expr<'a>),
    SubAssign(Expr<'a>),
    Assign(Expr<'a>),
}

impl<'a> AssignExpr<'a> {
    pub fn is_assign(&self) -> bool {
        matches!(self, Self::Assign(_))
    }
    pub fn inc() -> Self {
        Self::Inc
    }
    pub fn dec() -> Self {
        Self::Dec
    }
    pub fn add_assign(value: Expr<'a>) -> Self {
        Self::AddAssign(value)
    }
    pub fn sub_assign(value: Expr<'a>) -> Self {
        Self::SubAssign(value)
    }
    pub fn assign(value: Expr<'a>) -> Self {
        Self::Assign(value)
    }
}

#[derive(Debug, Clone)]
pub struct Assign<'a> {
    pub lhs: Location<'a>,
    pub op: AssignExpr<'a>,
    pub span: Span<'a>,
}

impl<'a> Assign<'a> {
    pub fn span(&self) -> Span<'a> {
        self.span
    }
    pub fn op(&self) -> &AssignExpr<'a> {
        &self.op
    }
    pub fn new(lhs: Location<'a>, rhs: AssignExpr<'a>, span: Span<'a>) -> Self {
        Self { lhs, op: rhs, span }
    }
    pub fn is_assign(&self) -> bool {
        self.op.is_assign()
    }
}

#[derive(Debug, Clone)]
pub enum Arg<'a> {
    String(Span<'a>),
    Expr(Expr<'a>),
}

#[derive(Debug, Clone, Copy)]
pub enum IntLiteral<'a> {
    Decimal(Span<'a>),
    Hex(Span<'a>),
}

impl<'a> IntLiteral<'a> {
    pub fn span(&self) -> Span<'a> {
        match self {
            Self::Decimal(span) => *span,
            Self::Hex(span) => *span,
        }
    }
}

impl<'a> From<IntLiteral<'a>> for Literal<'a> {
    fn from(value: IntLiteral<'a>) -> Self {
        match value {
            IntLiteral::Decimal(span) => Self::Decimal(span),
            IntLiteral::Hex(span) => Self::Hex(span),
        }
    }
}

impl<'a> TryFrom<Literal<'a>> for IntLiteral<'a> {
    type Error = ();
    fn try_from(value: Literal<'a>) -> Result<Self, Self::Error> {
        match value {
            Literal::Decimal(span) => Ok(Self::Decimal(span)),
            Literal::Hex(span) => Ok(Self::Hex(span)),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block<'a> {
    decls: Vec<PVar<'a>>,
    pub stmts: Vec<PStmt<'a>>,
}

impl<'a> Block<'a> {
    pub fn new() -> Self {
        Self {
            decls: Vec::new(),
            stmts: Vec::new(),
        }
    }
    pub fn decls(&self) -> &[PVar<'a>] {
        &self.decls
    }
    pub fn stmts(&self) -> &[PStmt<'a>] {
        &self.stmts
    }
    pub fn add(&mut self, elem: PBlockElem<'a>) {
        match elem {
            PBlockElem::Function(_) => {}
            PBlockElem::Decls(decls, ..) => self.decls.extend(decls),
            PBlockElem::Stmt(stmt) => self.stmts.push(stmt),
        }
    }
}

impl<'a> Default for Block<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// a literal that can be used as an expression
#[derive(Debug, Clone, Copy)]
pub enum Literal<'a> {
    Decimal(Span<'a>),
    Hex(Span<'a>),
    Char(char),
    Bool(bool),
}

impl From<char> for Literal<'_> {
    fn from(value: char) -> Self {
        Self::Char(value)
    }
}

impl From<bool> for Literal<'_> {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl<'a> Literal<'a> {
    pub fn hex(span: Span<'a>) -> Self {
        Self::Hex(span)
    }
    pub fn decimal(span: Span<'a>) -> Self {
        Self::Decimal(span)
    }
}

#[derive(Debug, Clone)]
pub enum Expr<'a> {
    Len {
        span: Span<'a>,
        id: Span<'a>,
    },
    Nested(Span<'a>, Box<Expr<'a>>),
    Not(Span<'a>, Box<Expr<'a>>),
    Neg(Span<'a>, Box<Expr<'a>>),
    Ter {
        cond: Box<Expr<'a>>,
        yes: Box<Expr<'a>>,
        no: Box<Expr<'a>>,
        span: Span<'a>,
    },
    Call(Call<'a>),
    Loc(Location<'a>),
    Literal {
        span: Span<'a>,
        value: Literal<'a>,
    },
    BinOp {
        op: Op,
        lhs: Box<Expr<'a>>,
        rhs: Box<Expr<'a>>,
        span: Span<'a>,
    },
}

impl<'a> From<Spanned<'a, Literal<'a>>> for Expr<'a> {
    fn from(value: Spanned<'a, Literal<'a>>) -> Self {
        let (value, span) = value.into_parts();
        Self::Literal { span, value }
    }
}

impl<'a> From<Spanned<'a, char>> for Expr<'a> {
    fn from(value: Spanned<'a, char>) -> Self {
        let (value, span) = value.into_parts();
        Self::Literal {
            span,
            value: value.into(),
        }
    }
}

impl<'a> From<Spanned<'a, bool>> for Expr<'a> {
    fn from(value: Spanned<'a, bool>) -> Self {
        let (value, span) = value.into_parts();
        Self::Literal {
            span,
            value: value.into(),
        }
    }
}

impl<'a> Expr<'a> {
    pub fn new_len(ident: Spanned<'a, Span<'a>>) -> Self {
        let (id, span) = ident.into_parts();
        Self::Len { span, id }
    }
    pub fn new_neg(expr: Spanned<'a, Expr<'a>>) -> Self {
        let (expr, span) = expr.into_parts();
        Self::Neg(span, Box::new(expr))
    }
    pub fn new_not(expr: Spanned<'a, Expr<'a>>) -> Self {
        let (expr, span) = expr.into_parts();
        Self::Not(span, Box::new(expr))
    }
    pub fn new_nested(expr: Spanned<'a, Expr<'a>>) -> Self {
        let (expr, span) = expr.into_parts();
        Self::Nested(span, Box::new(expr))
    }
    pub fn new_binop(lhs: Expr<'a>, rhs: Expr<'a>, op: Op, span: Span<'a>) -> Self {
        Self::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            span,
        }
    }
    pub fn new_ter(cond: Expr<'a>, yes: Expr<'a>, no: Expr<'a>, span: Span<'a>) -> Self {
        Self::Ter {
            cond: Box::new(cond),
            yes: Box::new(yes),
            no: Box::new(no),
            span,
        }
    }
    pub fn literal(&self) -> Option<&Literal<'a>> {
        match self {
            Self::Literal { value, .. } => Some(value),
            _ => None,
        }
    }
    pub fn span(&self) -> Span<'a> {
        match self {
            Self::Len { span, .. }
            | Self::Not(span, _)
            | Self::Neg(span, _)
            | Self::Literal { span, .. }
            | Self::Nested(span, _)
            | Self::Ter { span, .. }
            | Self::BinOp { span, .. } => *span,
            Self::Call(call) => call.span(),
            Self::Loc(l) => l.span(),
        }
    }
}

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

#[derive(Debug, Clone)]
pub enum PStmt<'a> {
    Call(Call<'a>),
    If {
        cond: Expr<'a>,
        yes: Block<'a>,
        no: Option<Block<'a>>,
        span: Span<'a>,
    },
    While {
        cond: Expr<'a>,
        body: Block<'a>,
        span: Span<'a>,
    },
    For {
        init: Assign<'a>,
        cond: Expr<'a>,
        update: Assign<'a>,
        body: Block<'a>,
        span: Span<'a>,
    },
    Assign(Assign<'a>),
    Return {
        expr: Option<Expr<'a>>,
        span: Span<'a>,
    },
    Break(Span<'a>),
    Continue(Span<'a>),
}

impl<'a> From<Assign<'a>> for PStmt<'a> {
    fn from(value: Assign<'a>) -> Self {
        Self::Assign(value)
    }
}

impl<'a> PStmt<'a> {
    pub fn r#break(span: Span<'a>) -> Self {
        Self::Break(span)
    }
    pub fn r#continue(span: Span<'a>) -> Self {
        Self::Continue(span)
    }
    pub fn r#if(cond: Expr<'a>, yes: Block<'a>, no: Option<Block<'a>>, span: Span<'a>) -> Self {
        Self::If {
            cond,
            yes,
            no,
            span,
        }
    }
    pub fn r#while(cond: Expr<'a>, body: Block<'a>, span: Span<'a>) -> Self {
        Self::While { cond, body, span }
    }
    pub fn r#for(
        init: Assign<'a>,
        cond: Expr<'a>,
        update: Assign<'a>,
        body: Block<'a>,
        span: Span<'a>,
    ) -> Self {
        Self::For {
            init,
            cond,
            update,
            body,
            span,
        }
    }
    pub fn r#return(expr: Option<Expr<'a>>, span: Span<'a>) -> Self {
        Self::Return { expr, span }
    }
    pub fn span(&self) -> Span<'a> {
        match self {
            Self::Call(call) => call.span(),
            Self::If { span, .. } => *span,
            Self::While { span, .. } => *span,
            Self::For { span, .. } => *span,
            Self::Assign(assign) => assign.span(),
            Self::Return { span, .. } => *span,
            Self::Break(span) => *span,
            Self::Continue(span) => *span,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PVar<'a> {
    Array {
        ty: Type,
        ident: Span<'a>,
        size: IntLiteral<'a>,
        // we do not need to record spans for identifiers
        span: Span<'a>,
    },
    Scalar {
        ty: Type,
        ident: Span<'a>,
    },
}

impl<'a> PVar<'a> {
    pub fn name(&self) -> Span<'a> {
        match self {
            Self::Array { ident, .. } => *ident,
            Self::Scalar { ident, .. } => *ident,
        }
    }
    pub fn r#type(&self) -> Type {
        match self {
            Self::Array { ty, .. } => *ty,
            Self::Scalar { ty, .. } => *ty,
        }
    }
    pub fn span(&self) -> Span<'a> {
        match self {
            Self::Array { span, .. } => *span,
            Self::Scalar { ident, .. } => *ident,
        }
    }
    pub fn new(ty: Type, ident: Span<'a>, size: Option<IntLiteral<'a>>, span: Span<'a>) -> Self {
        size.map_or_else(
            || Self::Scalar { ty, ident },
            |size| Self::Array {
                ty,
                ident,
                size,
                span,
            },
        )
    }
    pub fn ident(&self) -> Span<'a> {
        match self {
            Self::Array { ident, .. } => *ident,
            Self::Scalar { ident, .. } => *ident,
        }
    }
    pub fn scalar(ty: Type, ident: Span<'a>) -> Self {
        Self::Scalar { ty, ident }
    }
}

#[derive(Debug, Clone)]
pub enum Location<'a> {
    Scalar(Span<'a>),
    Index {
        ident: Span<'a>,
        offset: Box<Expr<'a>>,
        span: Span<'a>,
    },
}

impl<'a> Location<'a> {
    pub fn ident(&self) -> Span<'a> {
        match self {
            Location::Scalar(name) => *name,
            Location::Index { ident, .. } => *ident,
        }
    }
    pub fn span(&self) -> Span<'a> {
        match self {
            Location::Scalar(span) => *span,
            Location::Index { span, .. } => *span,
        }
    }
    pub fn new(ident: Span<'a>, offset: Option<Expr<'a>>, span: Span<'a>) -> Self {
        match offset {
            None => Location::Scalar(ident),
            Some(offset) => Location::Index {
                ident,
                offset: Box::new(offset),
                span,
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Call<'a> {
    pub name: Span<'a>,
    pub args: Vec<Arg<'a>>,
    pub span: Span<'a>,
}

impl<'a> Call<'a> {
    pub fn new(name: Span<'a>, args: Vec<Arg<'a>>, span: Span<'a>) -> Self {
        Self { name, args, span }
    }
    pub fn span(&self) -> Span<'a> {
        self.span
    }
}

impl<'a> From<Call<'a>> for PStmt<'a> {
    fn from(call: Call<'a>) -> Self {
        Self::Call(call)
    }
}

impl<'a> From<Call<'a>> for Expr<'a> {
    fn from(value: Call<'a>) -> Self {
        Self::Call(value)
    }
}

pub type PArgs<'a> = Vec<Arg<'a>>;

#[derive(Debug, Clone)]
pub struct PFunction<'a> {
    pub name: Span<'a>,
    pub body: Block<'a>,
    pub args: Vec<PVar<'a>>,
    pub ret: Option<Type>,
    span: Span<'a>,
}

impl<'a> PFunction<'a> {
    pub fn span(&self) -> Span<'a> {
        self.span
    }
    pub fn name(&self) -> Span<'a> {
        self.name
    }
    pub fn new(
        ret: Option<Type>,
        name: Span<'a>,
        args: Vec<PVar<'a>>,
        body: Block<'a>,
        span: Span<'a>,
    ) -> Self {
        Self {
            name,
            body,
            args,
            ret,
            span,
        }
    }
}

pub enum PBlockElem<'a> {
    Function(PFunction<'a>),
    Stmt(PStmt<'a>),
    Decls(Vec<PVar<'a>>, Span<'a>),
}

impl<'a> PBlockElem<'a> {
    pub fn span(&self) -> Span<'a> {
        match self {
            Self::Function(func) => func.span(),
            Self::Stmt(stmt) => stmt.span(),
            Self::Decls(_, span) => *span,
        }
    }
    pub fn decls(decls: Vec<PVar<'a>>, span: Span<'a>) -> Self {
        Self::Decls(decls, span)
    }
}

impl<'a> From<PStmt<'a>> for PBlockElem<'a> {
    fn from(stmt: PStmt<'a>) -> Self {
        Self::Stmt(stmt)
    }
}

impl<'a> From<PFunction<'a>> for PBlockElem<'a> {
    fn from(func: PFunction<'a>) -> Self {
        Self::Function(func)
    }
}

#[derive(Debug, Clone)]
pub enum PDocElem<'a> {
    Function(PFunction<'a>),
    Decl(Vec<PVar<'a>>, Span<'a>),
    Import(Import<'a>),
}

impl<'a> PDocElem<'a> {
    pub fn span(&self) -> Span<'a> {
        match self {
            Self::Function(func) => func.span,
            Self::Decl(_, span) => *span,
            Self::Import(import) => import.span,
        }
    }
}

impl<'a> PDocElem<'a> {
    pub fn decls(decls: Vec<PVar<'a>>, span: Span<'a>) -> Self {
        Self::Decl(decls, span)
    }
}

impl<'a> From<PFunction<'a>> for PDocElem<'a> {
    fn from(value: PFunction<'a>) -> Self {
        Self::Function(value)
    }
}

impl<'a> From<Import<'a>> for PDocElem<'a> {
    fn from(value: Import<'a>) -> Self {
        Self::Import(value)
    }
}

#[derive(Debug, Clone)]
pub struct PRoot<'a> {
    pub imports: Vec<Import<'a>>,
    pub decls: Vec<PVar<'a>>,
    pub funcs: Vec<PFunction<'a>>,
}

impl<'a> FromIterator<PDocElem<'a>> for PRoot<'a> {
    fn from_iter<T: IntoIterator<Item = PDocElem<'a>>>(iter: T) -> Self {
        iter.into_iter().fold(
            Self {
                decls: vec![],
                imports: vec![],
                funcs: vec![],
            },
            |mut root, elem| {
                match elem {
                    PDocElem::Decl(decls, _) => root.decls.extend(decls),
                    PDocElem::Function(func) => root.funcs.push(func),
                    PDocElem::Import(import) => root.imports.push(import),
                };
                root
            },
        )
    }
}

pub(super) mod checker {

    use super::*;
    use crate::parser::*;

    #[derive(Debug, Clone, PartialEq, Eq, Default)]
    pub struct BlockChecker<'a> {
        decls_finished: bool,
        decls_next_pos: Option<Span<'a>>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct StmtChecker;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct RootChecker<'a> {
        imports_finised: bool,
        imports_next_pos: Option<Span<'a>>,
        decls_finished: bool,
        decls_next_pos: Option<Span<'a>>,
    }

    impl StmtChecker {
        pub fn check<'a, F: FnMut(Error<'a>)>(stmt: &PStmt<'a>, mut callback: F) {
            let mut check_nested_expr = |e: &Expr<'a>| {
                if let Expr::Nested(..) = e {
                } else {
                    callback(Error::WrapInParens(e.span()))
                }
            };
            match stmt {
                PStmt::If { cond, .. } => check_nested_expr(cond),
                PStmt::While { cond, .. } => check_nested_expr(cond),
                PStmt::For { init, update, .. } => {
                    if let AssignExpr::Assign(..) = init.op {
                    } else {
                        callback(Error::ForInitHasToBeAssign(init.span()))
                    }
                    if let AssignExpr::Assign(..) = update.op {
                        callback(Error::ForUpdateIsIncOrCompound(update.span()))
                    } else {
                    }
                }
                PStmt::Return { .. }
                | PStmt::Continue(..)
                | PStmt::Break(..)
                | PStmt::Call(..)
                | PStmt::Assign(..) => {}
            }
        }
    }

    impl<'a> BlockChecker<'a> {
        pub fn new() -> Self {
            BlockChecker {
                decls_finished: false,
                decls_next_pos: None,
            }
        }

        pub fn check<F: FnMut(Error<'a>)>(&mut self, elem: &PBlockElem<'a>, mut callback: F) {
            match elem {
                PBlockElem::Decls(..) if self.decls_finished => {
                    assert!(self.decls_next_pos.is_some());
                    callback(Error::DeclAfterFunc {
                        decl_pos: elem.span(),
                        hinted_pos: self.decls_next_pos.unwrap(),
                    })
                }
                PBlockElem::Decls(..) => {}
                PBlockElem::Stmt(stmt) => {
                    if !self.decls_finished {
                        self.decls_finished = true;
                        self.decls_next_pos = Some(stmt.span());
                    }
                }
                PBlockElem::Function(..) => {}
            }
        }
    }

    impl Default for RootChecker<'_> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<'a> RootChecker<'a> {
        pub fn new() -> Self {
            RootChecker {
                imports_finised: false,
                imports_next_pos: None,
                decls_finished: false,
                decls_next_pos: None,
            }
        }

        pub fn check<F: FnMut(Error<'a>)>(&mut self, elem: &PDocElem<'a>, mut callback: F) {
            match elem {
                PDocElem::Import(import) if self.imports_finised => {
                    assert!(self.imports_next_pos.is_some());
                    let error = if self.decls_finished {
                        Error::ImportAfterDecl {
                            import_pos: import.span(),
                            hinted_pos: self.imports_next_pos.unwrap(),
                        }
                    } else {
                        Error::ImportAfterFunc {
                            import_pos: import.span(),
                            hinted_pos: self.imports_next_pos.unwrap(),
                        }
                    };
                    callback(error)
                }
                PDocElem::Import(..) => {}
                PDocElem::Decl(_, span) if self.decls_finished => {
                    assert!(self.decls_next_pos.is_some());
                    callback(Error::DeclAfterFunc {
                        decl_pos: *span,
                        hinted_pos: self.decls_next_pos.unwrap(),
                    })
                }
                PDocElem::Decl(..) => {
                    if !self.imports_finised {
                        self.imports_finised = true;
                        self.imports_next_pos = Some(elem.span());
                    }
                    {}
                }
                PDocElem::Function(..) => {
                    if !self.imports_finised {
                        self.imports_finised = true;
                        self.imports_next_pos = Some(elem.span());
                    }
                    if !self.decls_finished {
                        self.decls_finished = true;
                        self.decls_next_pos = Some(elem.span());
                    }
                }
            }
        }
    }
}
