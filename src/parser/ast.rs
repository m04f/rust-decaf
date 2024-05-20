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
pub struct PImport<'a> {
    span: Span<'a>,
    id: Span<'a>,
}

impl<'a> PImport<'a> {
    pub fn name(&self) -> Span<'a> {
        self.id
    }
    pub fn span(&self) -> Span<'a> {
        self.span
    }
}

impl<'a> From<Spanned<'a, Span<'a>>> for PImport<'a> {
    fn from(value: Spanned<'a, Span<'a>>) -> Self {
        let (ident, span) = value.into_parts();
        Self { span, id: ident }
    }
}

#[derive(Debug, Clone)]
pub struct PString<'a>(Span<'a>);

impl<'a> PString<'a> {
    pub fn span(&self) -> Span<'a> {
        self.0
    }
}

impl<'a> From<Span<'a>> for PString<'a> {
    fn from(value: Span<'a>) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone)]
pub enum PAssignExpr<'a> {
    Inc,
    Dec,
    AddAssign(PExpr<'a>),
    SubAssign(PExpr<'a>),
    Assign(PExpr<'a>),
}

impl<'a> PAssignExpr<'a> {
    pub fn is_assign(&self) -> bool {
        matches!(self, Self::Assign(_))
    }
    pub fn inc() -> Self {
        Self::Inc
    }
    pub fn dec() -> Self {
        Self::Dec
    }
    pub fn add_assign(value: PExpr<'a>) -> Self {
        Self::AddAssign(value)
    }
    pub fn sub_assign(value: PExpr<'a>) -> Self {
        Self::SubAssign(value)
    }
    pub fn assign(value: PExpr<'a>) -> Self {
        Self::Assign(value)
    }
}

#[derive(Debug, Clone)]
pub struct PAssign<'a> {
    pub lhs: PLoc<'a>,
    pub op: PAssignExpr<'a>,
    pub span: Span<'a>,
}

impl<'a> PAssign<'a> {
    pub fn span(&self) -> Span<'a> {
        self.span
    }
    pub fn op(&self) -> &PAssignExpr<'a> {
        &self.op
    }
    pub fn new(lhs: PLoc<'a>, rhs: PAssignExpr<'a>, span: Span<'a>) -> Self {
        Self { lhs, op: rhs, span }
    }
    pub fn is_assign(&self) -> bool {
        self.op.is_assign()
    }
}

#[derive(Debug, Clone)]
pub enum PArg<'a> {
    String(PString<'a>),
    Expr(PExpr<'a>),
}

impl<'a> From<PString<'a>> for PArg<'a> {
    fn from(s: PString<'a>) -> Self {
        Self::String(s)
    }
}

impl<'a> From<PExpr<'a>> for PArg<'a> {
    fn from(value: PExpr<'a>) -> Self {
        Self::Expr(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PIntLiteral<'a> {
    Decimal(Span<'a>),
    Hex(Span<'a>),
}

impl<'a> PIntLiteral<'a> {
    pub fn span(&self) -> Span<'a> {
        match self {
            Self::Decimal(span) => *span,
            Self::Hex(span) => *span,
        }
    }
}

impl<'a> From<PIntLiteral<'a>> for PLiteral<'a> {
    fn from(value: PIntLiteral<'a>) -> Self {
        match value {
            PIntLiteral::Decimal(span) => Self::Decimal(span),
            PIntLiteral::Hex(span) => Self::Hex(span),
        }
    }
}

impl<'a> TryFrom<PLiteral<'a>> for PIntLiteral<'a> {
    type Error = ();
    fn try_from(value: PLiteral<'a>) -> Result<Self, Self::Error> {
        match value {
            PLiteral::Decimal(span) => Ok(Self::Decimal(span)),
            PLiteral::Hex(span) => Ok(Self::Hex(span)),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PBlock<'a> {
    decls: Vec<PVar<'a>>,
    pub stmts: Vec<PStmt<'a>>,
}

impl<'a> PBlock<'a> {
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

impl<'a> Default for PBlock<'a> {
    fn default() -> Self {
        Self::new()
    }
}

/// a literal that can be used as an expression
#[derive(Debug, Clone, Copy)]
pub enum PLiteral<'a> {
    Decimal(Span<'a>),
    Hex(Span<'a>),
    Char(char),
    Bool(bool),
}

impl From<char> for PLiteral<'_> {
    fn from(value: char) -> Self {
        Self::Char(value)
    }
}

impl From<bool> for PLiteral<'_> {
    fn from(value: bool) -> Self {
        Self::Bool(value)
    }
}

impl<'a> PLiteral<'a> {
    pub fn hex(span: Span<'a>) -> Self {
        Self::Hex(span)
    }
    pub fn decimal(span: Span<'a>) -> Self {
        Self::Decimal(span)
    }
}

#[derive(Debug, Clone)]
pub enum PExpr<'a> {
    Len {
        span: Span<'a>,
        id: Span<'a>,
    },
    Nested(Span<'a>, Box<PExpr<'a>>),
    Not(Span<'a>, Box<PExpr<'a>>),
    Neg(Span<'a>, Box<PExpr<'a>>),
    Ter {
        cond: Box<PExpr<'a>>,
        yes: Box<PExpr<'a>>,
        no: Box<PExpr<'a>>,
        span: Span<'a>,
    },
    Call(PCall<'a>),
    Index {
        name: Span<'a>,
        offset: Box<PExpr<'a>>,
        span: Span<'a>,
    },

    Scalar(Span<'a>),
    Literal {
        span: Span<'a>,
        value: PLiteral<'a>,
    },
    BinOp {
        op: Op,
        lhs: Box<PExpr<'a>>,
        rhs: Box<PExpr<'a>>,
        span: Span<'a>,
    },
}

impl<'a> From<Spanned<'a, PLiteral<'a>>> for PExpr<'a> {
    fn from(value: Spanned<'a, PLiteral<'a>>) -> Self {
        let (value, span) = value.into_parts();
        Self::Literal { span, value }
    }
}

impl<'a> From<Spanned<'a, char>> for PExpr<'a> {
    fn from(value: Spanned<'a, char>) -> Self {
        let (value, span) = value.into_parts();
        Self::Literal {
            span,
            value: value.into(),
        }
    }
}

impl<'a> From<Spanned<'a, bool>> for PExpr<'a> {
    fn from(value: Spanned<'a, bool>) -> Self {
        let (value, span) = value.into_parts();
        Self::Literal {
            span,
            value: value.into(),
        }
    }
}

impl<'a> From<PLoc<'a>> for PExpr<'a> {
    fn from(value: PLoc<'a>) -> Self {
        value.offset.map_or_else(
            || Self::Scalar(value.ident),
            |offset| Self::Index {
                name: value.ident,
                offset: Box::new(offset),
                span: value.span,
            },
        )
    }
}

impl<'a> PExpr<'a> {
    pub fn new_len(ident: Spanned<'a, Span<'a>>) -> Self {
        let (id, span) = ident.into_parts();
        Self::Len { span, id }
    }
    pub fn new_neg(expr: Spanned<'a, PExpr<'a>>) -> Self {
        let (expr, span) = expr.into_parts();
        Self::Neg(span, Box::new(expr))
    }
    pub fn new_not(expr: Spanned<'a, PExpr<'a>>) -> Self {
        let (expr, span) = expr.into_parts();
        Self::Not(span, Box::new(expr))
    }
    pub fn new_nested(expr: Spanned<'a, PExpr<'a>>) -> Self {
        let (expr, span) = expr.into_parts();
        Self::Nested(span, Box::new(expr))
    }
    pub fn new_binop(lhs: PExpr<'a>, rhs: PExpr<'a>, op: Op, span: Span<'a>) -> Self {
        Self::BinOp {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            span,
        }
    }
    pub fn new_ter(cond: PExpr<'a>, yes: PExpr<'a>, no: PExpr<'a>, span: Span<'a>) -> Self {
        Self::Ter {
            cond: Box::new(cond),
            yes: Box::new(yes),
            no: Box::new(no),
            span,
        }
    }
    pub fn literal(&self) -> Option<&PLiteral<'a>> {
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
            | Self::Literal { span, .. } => *span,
            Self::Nested(span, _) => *span,
            Self::Scalar(ident) => *ident,
            Self::Ter { span, .. } => *span,
            Self::Call(call) => call.span(),
            Self::Index { span, .. } => *span,
            Self::BinOp { span, .. } => *span,
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
    Call(PCall<'a>),
    If {
        cond: PExpr<'a>,
        yes: PBlock<'a>,
        no: Option<PBlock<'a>>,
        span: Span<'a>,
    },
    While {
        cond: PExpr<'a>,
        body: PBlock<'a>,
        span: Span<'a>,
    },
    For {
        init: PAssign<'a>,
        cond: PExpr<'a>,
        update: PAssign<'a>,
        body: PBlock<'a>,
        span: Span<'a>,
    },
    Assign(PAssign<'a>),
    Return {
        expr: Option<PExpr<'a>>,
        span: Span<'a>,
    },
    Break(Span<'a>),
    Continue(Span<'a>),
}

impl<'a> From<PAssign<'a>> for PStmt<'a> {
    fn from(value: PAssign<'a>) -> Self {
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
    pub fn r#if(cond: PExpr<'a>, yes: PBlock<'a>, no: Option<PBlock<'a>>, span: Span<'a>) -> Self {
        Self::If {
            cond,
            yes,
            no,
            span,
        }
    }
    pub fn r#while(cond: PExpr<'a>, body: PBlock<'a>, span: Span<'a>) -> Self {
        Self::While { cond, body, span }
    }
    pub fn r#for(
        init: PAssign<'a>,
        cond: PExpr<'a>,
        update: PAssign<'a>,
        body: PBlock<'a>,
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
    pub fn r#return(expr: Option<PExpr<'a>>, span: Span<'a>) -> Self {
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
        size: PIntLiteral<'a>,
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
    pub fn new(ty: Type, ident: Span<'a>, size: Option<PIntLiteral<'a>>, span: Span<'a>) -> Self {
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
pub struct PLoc<'a> {
    pub ident: Span<'a>,
    pub offset: Option<PExpr<'a>>,
    span: Span<'a>,
}

impl<'a> PLoc<'a> {
    pub fn ident(&self) -> Span<'a> {
        self.ident
    }
    pub fn span(&self) -> Span<'a> {
        self.span
    }
    pub fn new(ident: Span<'a>, offset: Option<PExpr<'a>>, span: Span<'a>) -> Self {
        Self {
            ident,
            offset,
            span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PCall<'a> {
    pub name: Span<'a>,
    pub args: Vec<PArg<'a>>,
    pub span: Span<'a>,
}

impl<'a> PCall<'a> {
    pub fn new(name: Span<'a>, args: Vec<PArg<'a>>, span: Span<'a>) -> Self {
        Self { name, args, span }
    }
    pub fn span(&self) -> Span<'a> {
        self.span
    }
}

impl<'a> From<PCall<'a>> for PStmt<'a> {
    fn from(call: PCall<'a>) -> Self {
        Self::Call(call)
    }
}

impl<'a> From<PCall<'a>> for PExpr<'a> {
    fn from(value: PCall<'a>) -> Self {
        Self::Call(value)
    }
}

pub type PArgs<'a> = Vec<PArg<'a>>;

#[derive(Debug, Clone)]
pub struct PFunction<'a> {
    pub name: Span<'a>,
    pub body: PBlock<'a>,
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
        body: PBlock<'a>,
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
    Import(PImport<'a>),
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

impl<'a> From<PImport<'a>> for PDocElem<'a> {
    fn from(value: PImport<'a>) -> Self {
        Self::Import(value)
    }
}

#[derive(Debug, Clone)]
pub struct PRoot<'a> {
    pub imports: Vec<PImport<'a>>,
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
            let mut check_nested_expr = |e: &PExpr<'a>| {
                if let PExpr::Nested(..) = e {
                } else {
                    callback(Error::WrapInParens(e.span()))
                }
            };
            match stmt {
                PStmt::If { cond, .. } => check_nested_expr(cond),
                PStmt::While { cond, .. } => check_nested_expr(cond),
                PStmt::For { init, update, .. } => {
                    if let PAssignExpr::Assign(..) = init.op {
                    } else {
                        callback(Error::ForInitHasToBeAssign(init.span()))
                    }
                    if let PAssignExpr::Assign(..) = update.op {
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
