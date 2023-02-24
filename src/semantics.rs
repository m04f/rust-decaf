use crate::{
    ast::{self, ELiteral, Loc, StringLiteral, Type, Var},
    parser::*,
    span::*,
};

use std::collections::HashMap;

#[derive(Debug)]
pub enum Error<'a> {
    UndeclaredIdentifier(Span<'a>),
    ExpectedArrayVariable(Span<'a>),
    ExpectedScalarVariable(Span<'a>),
    CannotIndexScalar(Span<'a>),
    CannotAssignToArray(Span<'a>),
    ExpectedBoolExpr(Span<'a>),
    ExpectedIntExpr(Span<'a>),
    ReturnValueFromVoid(Span<'a>),
    Redifinition(Span<'a>, Span<'a>),
    BreakOutsideLoop(Span<'a>),
    ContinueOutsideLoop(Span<'a>),
    VoidFuncAsExpr(Span<'a>),
    TypeMismatch {
        lhs: Type,
        rhs: Type,
        lspan: Span<'a>,
        rspan: Span<'a>,
    },
    WrongNumberOfArgs {
        expected: usize,
        found: usize,
        span: Span<'a>,
    },
    ExpectedType(Type, Span<'a>),
    ExpectedExpression(Span<'a>),
    ZeroArraySize(Span<'a>),
    TooLargeInt(Span<'a>),
    RootDoesNotContainMain,
    InvalidMainSig(Span<'a>),
    VariableNotAMethod(Span<'a>),
}

impl ast::Op {
    fn is_arith(&self) -> bool {
        matches!(
            self,
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Mod
        )
    }
    fn is_relop(&self) -> bool {
        matches!(
            self,
            Self::Less | Self::LessEqual | Self::Greater | Self::GreaterEqual
        )
    }
    fn is_eq(&self) -> bool {
        matches!(self, Self::Equal | Self::NotEqual)
    }
    fn is_cond(&self) -> bool {
        matches!(self, Self::And | Self::Or)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HIRLiteral {
    Int(i64),
    Bool(bool),
}

#[derive(Debug)]
struct SymTable<'a, 'b, O> {
    map: &'b HashMap<Span<'a>, O>,
    parent: Option<&'b Self>,
}

impl<'a, 'b, O> Clone for SymTable<'a, 'b, O> {
    fn clone(&self) -> Self {
        Self {
            map: self.map,
            parent: self.parent,
        }
    }
}

impl<'a, 'b, O> Copy for SymTable<'a, 'b, O> {}

type FSymMap<'a, 'b> = SymTable<'a, 'b, FunctionSig<'a>>;
type VSymMap<'a, 'b> = SymTable<'a, 'b, Var<Span<'a>>>;

impl<'a, 'b, O> SymTable<'a, 'b, O> {
    fn new(map: &'b HashMap<Span<'a>, O>) -> Self {
        Self { map, parent: None }
    }

    fn parent(self, parent: &'b Self) -> Self {
        Self {
            map: self.map,
            parent: Some(parent),
        }
    }

    fn get_sym(&self, sym: Span<'a>) -> Option<&O> {
        if let Some(parent) = self.parent {
            self.map.get(&sym).or_else(|| parent.get_sym(sym))
        } else {
            self.map.get(&sym)
        }
    }
}

impl HIRLiteral {
    fn int(self) -> Option<i64> {
        match self {
            Self::Int(val) => Some(val),
            _ => None,
        }
    }

    fn ty(self) -> Type {
        match self {
            Self::Int(_) => Type::Int,
            Self::Bool(_) => Type::Bool,
        }
    }

    fn from_pliteral(literal: ELiteral<Span<'_>>, is_neg: bool) -> Result<Self, Vec<Error<'_>>> {
        let map_digit = if is_neg { |dig: i64| -dig } else { |dig| dig };
        match literal {
            ast::ELiteral::Decimal(num) => {
                let parsed = num.bytes().try_fold(0, |acc: i64, digit| {
                    acc.checked_mul(10)
                        .and_then(|acc| acc.checked_add(map_digit((digit - b'0') as i64)))
                });
                parsed
                    .ok_or(vec![Error::TooLargeInt(num)])
                    .map(HIRLiteral::Int)
            }
            ast::ELiteral::Hex(num) => num
                .bytes()
                .try_fold(0, |acc: i64, digit| match digit {
                    b'0'..=b'9' => acc
                        .checked_mul(16)
                        .and_then(|acc| acc.checked_add(map_digit((digit - b'0') as i64))),
                    b'a'..=b'f' => acc
                        .checked_mul(16)
                        .and_then(|acc| acc.checked_add(map_digit((digit - b'a' + 10) as i64))),
                    b'A'..=b'F' => acc
                        .checked_mul(16)
                        .and_then(|acc| acc.checked_add(map_digit((digit - b'A' + 10) as i64))),
                    _ => {
                        unreachable!()
                    }
                })
                .ok_or(vec![Error::TooLargeInt(num)])
                .map(HIRLiteral::Int),
            ast::ELiteral::Bool(val) => Ok(HIRLiteral::Bool(val)),
            ast::ELiteral::Char(c) => Ok(HIRLiteral::Int(c.into())),
        }
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
    fn name(&self) -> Span<'a> {
        match self {
            Self::Extern(name) => *name,
            Self::Decl { name, .. } => *name,
        }
    }
    fn from_pfunction(func: &PFunction<'a>) -> Self {
        Self::Decl {
            name: *func.name.span(),
            arg_types: func.args.iter().map(|arg| arg.ty()).collect(),
            ty: func.ret,
        }
    }
    fn from_pimport(import: &PImport<'a>) -> Self {
        Self::Extern(*import.name().span())
    }
}

impl FunctionSig<'_> {
    fn ty(&self) -> Option<Type> {
        if let Self::Decl { ty, .. } = self {
            *ty
        } else {
            Some(Type::Int)
        }
    }
}

pub type SymMap<'a, T> = HashMap<Span<'a>, T>;
pub type VarSymMap<'a> = SymMap<'a, Var<Span<'a>>>;
pub type FuncSymMap<'a> = SymMap<'a, HIRFunction<'a>>;
pub type ImportSymMap<'a> = SymMap<'a, HIRImport<'a>>;
pub type SigSymMap<'a> = SymMap<'a, FunctionSig<'a>>;
pub type HIRExpr<'a> = ast::Expr<HIRLiteral, HIRArg<'a>, Span<'a>, Type>;
#[derive(Debug, Clone)]
pub enum HIRArg<'a> {
    String(StringLiteral<Span<'a>>),
    Array(Span<'a>),
    Expr(HIRExpr<'a>),
}
pub type HIRCall<'a> = ast::Call<Span<'a>, HIRArg<'a>>;
pub type HIRStmt<'a> = ast::Stmt<HIRLiteral, HIRArg<'a>, Span<'a>, VarSymMap<'a>, Type>;
pub type HIRBlock<'a> = ast::Block<HIRLiteral, HIRArg<'a>, Span<'a>, VarSymMap<'a>, Type>;
pub type StmtBlock<'a> = HIRBlock<'a>;
pub type FuncBlock<'a> = HIRBlock<'a>;
pub type HIRAssign<'a> = ast::Assign<HIRLiteral, HIRArg<'a>, Span<'a>, Type>;
pub type HIRAssignExpr<'a> = ast::AssignExpr<HIRLiteral, HIRArg<'a>, Span<'a>, Type>;
pub type HIRFunction<'a> = ast::Function<HIRBlock<'a>, VarSymMap<'a>, Span<'a>>;
pub type HIRImport<'a> = ast::Import<Span<'a>>;

pub struct HIRRoot<'a> {
    globals: VarSymMap<'a>,
    functions: FuncSymMap<'a>,
    imports: ImportSymMap<'a>,
}

impl<'a> HIRArg<'a> {
    fn span(&self) -> &Span<'a> {
        match self {
            Self::String(string) => string.span(),
            Self::Array(span) => span,
            Self::Expr(expr) => expr.span(),
        }
    }
}

impl<'a> HIRExpr<'a> {
    fn len(span: Span<'a>, ident: ast::Identifier<Span<'a>>) -> Self {
        let ty = Type::Int;
        Self {
            inner: ast::ExprInner::Len(span, ident),
            extra: ty,
        }
    }

    fn not(self, span: Span<'a>) -> Self {
        let ty = self.extra;
        assert!(ty == Type::Bool);
        Self {
            inner: ast::ExprInner::Not(span, Box::new(self)),
            extra: ty,
        }
    }

    fn neg(self, span: Span<'a>) -> Self {
        let ty = self.extra;
        assert!(ty == Type::Int);
        Self {
            inner: ast::ExprInner::Neg(span, Box::new(self)),
            extra: ty,
        }
    }

    fn nested(self, span: Span<'a>) -> Self {
        let ty = self.extra;
        Self {
            inner: ast::ExprInner::Nested(span, Box::new(self)),
            extra: ty,
        }
    }

    fn ty(&self) -> Type {
        self.extra
    }

    fn is_boolean(&self) -> bool {
        self.extra == Type::Bool
    }

    fn is_int(&self) -> bool {
        self.extra == Type::Int
    }

    fn from_pexpr(
        expr: PExpr<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        use ast::ExprInner::*;
        match expr.inner {
            Len(span, ident) => match vst.get_sym(*ident.span()) {
                None => Err(vec![Error::UndeclaredIdentifier(*ident.span())]),
                Some(var) => match var {
                    Var::Scalar { .. } => Err(vec![Error::ExpectedArrayVariable(*ident.span())]),
                    _ => Ok(Self::len(span, ident)),
                },
            },
            Not(span, e) => {
                let e = Self::from_pexpr(*e, vst, fst)?;
                if !e.is_boolean() {
                    Err(vec![Error::ExpectedBoolExpr(span)])
                } else {
                    Ok(Self::not(e, span))
                }
            }
            Neg(span, e) if e.is_intliteral() => {
                HIRLiteral::from_pliteral(e.literal().unwrap(), true).map(|value| Self {
                    inner: Literal { span, value },
                    extra: Type::Int,
                })
            }
            Neg(span, e) => {
                let e = Self::from_pexpr(*e, vst, fst)?;
                if !e.is_int() {
                    Err(vec![Error::ExpectedIntExpr(span)])
                } else {
                    Ok(Self::neg(e, span))
                }
            }
            Nested(span, e) => {
                let e = Self::from_pexpr(*e, vst, fst)?;
                Ok(Self::nested(e, span))
            }
            Index { name, offset, span } => match vst.get_sym(*name.span()) {
                None => Err(vec![Error::UndeclaredIdentifier(*name.span())]),
                Some(Var::Scalar { .. }) => Err(vec![Error::ExpectedArrayVariable(*name.span())]),
                Some(var) => {
                    let index = Self::from_pexpr(*offset, vst, fst)?;
                    if !index.is_int() {
                        Err(vec![Error::ExpectedIntExpr(span)])
                    } else {
                        Ok(Self {
                            extra: var.ty(),
                            inner: ast::ExprInner::Index {
                                name,
                                offset: Box::new(index),
                                span,
                            },
                        })
                    }
                }
            },
            Scalar(ident) => match vst.get_sym(*ident.span()) {
                None => Err(vec![Error::UndeclaredIdentifier(*ident.span())]),
                Some(var) if var.is_scalar() => Ok(Self {
                    extra: var.ty(),
                    inner: ast::ExprInner::Scalar(ident),
                }),
                _ => Err(vec![Error::ExpectedScalarVariable(*ident.span())]),
            },
            Ter {
                cond,
                yes,
                no,
                span,
            } => {
                let cond = Self::from_pexpr(*cond, vst, fst);
                let yes = Self::from_pexpr(*yes, vst, fst);
                let no = Self::from_pexpr(*no, vst, fst);
                if cond.is_ok() && yes.is_ok() && no.is_ok() {
                    let cond = cond.unwrap();
                    let yes = yes.unwrap();
                    let no = no.unwrap();
                    if cond.is_boolean() && yes.ty() == no.ty() {
                        Ok(Self {
                            extra: yes.ty(),
                            inner: ast::ExprInner::Ter {
                                cond: Box::new(cond),
                                yes: Box::new(yes),
                                no: Box::new(no),
                                span,
                            },
                        })
                    } else {
                        let mut errors = vec![];
                        (!cond.is_boolean())
                            .then(|| errors.push(Error::ExpectedBoolExpr(*cond.span())));
                        (yes.ty() != no.ty()).then(|| {
                            errors.push(Error::TypeMismatch {
                                rspan: *yes.span(),
                                rhs: yes.ty(),
                                lspan: *no.span(),
                                lhs: no.ty(),
                            })
                        });
                        Err(errors)
                    }
                } else {
                    let mut errors = vec![];
                    if let Some(e) = cond.err() {
                        errors.extend(e)
                    }
                    if let Some(e) = no.err() {
                        errors.extend(e)
                    }
                    if let Some(e) = yes.err() {
                        errors.extend(e)
                    }
                    Err(errors)
                }
            }
            BinOp { op, lhs, rhs, span } => {
                let lhs = Self::from_pexpr(*lhs, vst, fst);
                let rhs = Self::from_pexpr(*rhs, vst, fst);
                if lhs.is_ok() && rhs.is_ok() {
                    let lhs = lhs.unwrap();
                    let rhs = rhs.unwrap();
                    if op.is_eq() {
                        if lhs.ty() == rhs.ty() {
                            Ok(Self {
                                extra: Type::Bool,
                                inner: ast::ExprInner::BinOp {
                                    op,
                                    lhs: Box::new(lhs),
                                    rhs: Box::new(rhs),
                                    span,
                                },
                            })
                        } else {
                            Err(vec![Error::TypeMismatch {
                                lspan: *lhs.span(),
                                lhs: lhs.ty(),
                                rspan: *rhs.span(),
                                rhs: rhs.ty(),
                            }])
                        }
                    } else if op.is_arith() || op.is_relop() {
                        if lhs.ty() == Type::Int && rhs.ty() == Type::Int {
                            Ok(Self {
                                extra: if op.is_arith() { Type::Int } else { Type::Bool },
                                inner: ast::ExprInner::BinOp {
                                    op,
                                    lhs: Box::new(lhs),
                                    rhs: Box::new(rhs),
                                    span,
                                },
                            })
                        } else {
                            let mut errors = vec![];
                            if lhs.ty() != Type::Int {
                                errors.push(Error::ExpectedIntExpr(*lhs.span()))
                            }
                            if rhs.ty() != Type::Int {
                                errors.push(Error::ExpectedIntExpr(*rhs.span()))
                            }
                            Err(errors)
                        }
                    } else if op.is_cond() {
                        if lhs.ty() == Type::Bool && rhs.ty() == Type::Bool {
                            Ok(Self {
                                extra: Type::Bool,
                                inner: ast::ExprInner::BinOp {
                                    op,
                                    lhs: Box::new(lhs),
                                    rhs: Box::new(rhs),
                                    span,
                                },
                            })
                        } else {
                            let mut errors = vec![];
                            if lhs.ty() != Type::Bool {
                                errors.push(Error::ExpectedBoolExpr(*lhs.span()))
                            }
                            if rhs.ty() != Type::Bool {
                                errors.push(Error::ExpectedBoolExpr(*rhs.span()))
                            }
                            Err(errors)
                        }
                    } else {
                        unreachable!()
                    }
                } else {
                    let mut errors = vec![];
                    if let Some(e) = lhs.err() {
                        errors.extend(e)
                    }
                    if let Some(e) = rhs.err() {
                        errors.extend(e)
                    }
                    Err(errors)
                }
            }
            Literal { span, value } => HIRLiteral::from_pliteral(value, false).map(|value| Self {
                inner: Literal { span, value },
                extra: value.ty(),
            }),
            Call(call) => {
                let name = *call.name.span();
                let call = HIRCall::from_pcall(call, vst, fst).map(Call)?;
                let ty = fst
                    .get_sym(name)
                    .unwrap()
                    .ty()
                    .ok_or(vec![Error::VoidFuncAsExpr(*call.span())])?;
                Ok(Self {
                    extra: ty,
                    inner: call,
                })
            }
        }
    }
}

impl<'a> HIRArg<'a> {
    fn extern_from_pcall(
        arg: PArg<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        match arg {
            PArg::Expr(PExpr { inner, .. })
                if let Some(ident) = inner
                    .scalar()
                    .and_then(|ident| {
                        vst.get_sym(*ident.span()).and_then(|var| var.is_array().then_some(*var.span()))
                    })
                    => {
                        Ok(Self::Array(ident))
                    }
            PArg::Expr(e) => HIRExpr::from_pexpr(e, vst, fst).map(Self::Expr),
            PArg::String(s) => Ok(Self::String(s)),
        }
    }

    fn ty(&self) -> Option<Type> {
        match self {
            Self::String(_) | Self::Array(_) => None,
            Self::Expr(e) => Some(e.ty()),
        }
    }

    fn user_defined_from_pcall(
        arg: PArg<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        match arg {
            PArg::String(string) => Err(vec![Error::ExpectedExpression(*string.span())]),
            PArg::Expr(e) => HIRExpr::from_pexpr(e, vst, fst).map(Self::Expr),
        }
    }
}

trait FoldResult<'a, R> {
    fn fold_result(self) -> Result<Vec<R>, Vec<Error<'a>>>;
}

impl<'a, T: Iterator<Item = Result<R, Vec<Error<'a>>>>, R> FoldResult<'a, R> for T {
    fn fold_result(self) -> Result<Vec<R>, Vec<Error<'a>>> {
        self.fold(Ok(vec![]), |acc, item| match (acc, item) {
            (Ok(mut acc), Ok(item)) => {
                acc.push(item);
                Ok(acc)
            }
            (Ok(_), Err(e)) | (Err(e), Ok(_)) => Err(e),
            (Err(mut e1), Err(e2)) => {
                e1.extend(e2);
                Err(e1)
            }
        })
    }
}

impl<'a> HIRCall<'a> {
    fn from_pcall(
        call: PCall<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        if vst.get_sym(*call.name.span()).is_some() {
            Err(vec![Error::VariableNotAMethod(*call.name.span())])
        } else {
            match fst.get_sym(*call.name.span()) {
                None => Err(vec![Error::UndeclaredIdentifier(*call.name.span())]),
                Some(FunctionSig::Extern(..)) => call
                    .args
                    .into_iter()
                    .map(|arg| HIRArg::extern_from_pcall(arg, vst, fst))
                    .fold_result()
                    .map(|args| HIRCall {
                        name: call.name,
                        args,
                        span: call.span,
                    }),
                Some(FunctionSig::Decl { arg_types, .. }) => {
                    if call.args.len() == arg_types.len() {
                        call.args
                            .into_iter()
                            .zip(arg_types.iter())
                            .map(|(arg, ty)| {
                                HIRArg::user_defined_from_pcall(arg, vst, fst).and_then(|arg| {
                                    if arg.ty().unwrap() != *ty {
                                        Err(vec![Error::ExpectedType(*ty, *arg.span())])
                                    } else {
                                        Ok(arg)
                                    }
                                })
                            })
                            .fold_result()
                            .map(|args| HIRCall {
                                name: call.name,
                                args,
                                span: call.span,
                            })
                    } else {
                        Err(vec![Error::WrongNumberOfArgs {
                            expected: arg_types.len(),
                            found: call.args.len(),
                            span: *call.name.span(),
                        }])
                    }
                }
            }
        }
    }
}

impl<'a> HIRAssignExpr<'a> {
    fn ty(&self) -> Type {
        use ast::AssignExpr::*;
        match self {
            Inc | Dec => Type::Int,
            AddAssign(e) | SubAssign(e) | Assign(e) => e.ty(),
        }
    }
}

impl<'a> HIRAssign<'a> {
    fn from_passign(
        assign: PAssign<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        use ast::AssignExpr::*;
        let op = match assign.op {
            Inc => Ok(Inc),
            Dec => Ok(Dec),
            AddAssign(e) => HIRExpr::from_pexpr(e, vst, fst).map(AddAssign),
            SubAssign(e) => HIRExpr::from_pexpr(e, vst, fst).map(SubAssign),
            Assign(e) => HIRExpr::from_pexpr(e, vst, fst).map(Assign),
        };
        let loc = match vst.get_sym(*assign.lhs.ident.span()) {
            Some(loc) => match loc {
                Var::Scalar { ident, .. } if assign.lhs.is_indexed() => {
                    Err(vec![Error::CannotIndexScalar(*ident.span())])
                }
                Var::Scalar { ty, ident } => Ok((ty, ast::Loc::from_ident(*ident))),
                Var::Array { ident, .. } if assign.lhs.is_scalar() => {
                    Err(vec![Error::CannotAssignToArray(*ident.span())])
                }
                Var::Array {
                    ty, ident, span, ..
                } => {
                    let index = assign.lhs.offset.unwrap();
                    HIRExpr::from_pexpr(index, vst, fst).and_then(|index| {
                        if index.ty() != Type::Int {
                            Err(vec![Error::ExpectedIntExpr(*index.span())])
                        } else {
                            Ok((ty, Loc::with_offset(*ident, index, *span)))
                        }
                    })
                }
            },
            None => Err(vec![Error::UndeclaredIdentifier(*assign.lhs.ident.span())]),
        };
        match (op, loc) {
            (Ok(Inc), Ok((&ty, loc))) if ty == Type::Int => Ok(Self::new(loc, Inc, assign.span)),
            (Ok(Inc), Ok((_, loc))) => Err(vec![Error::ExpectedIntExpr(*loc.span())]),
            (Ok(Dec), Ok((&ty, loc))) if ty == Type::Int => Ok(Self::new(loc, Dec, assign.span)),
            (Ok(Dec), Ok((_, loc))) => Err(vec![Error::ExpectedIntExpr(*loc.span())]),
            (Ok(AddAssign(e)), Ok((&ty, loc))) if ty == Type::Int && e.ty() == Type::Int => {
                Ok(Self::new(loc, AddAssign(e), assign.span))
            }
            (Ok(AddAssign(_)), Ok((&ty, loc))) if ty != Type::Int => {
                Err(vec![Error::ExpectedIntExpr(*loc.span())])
            }
            (Ok(AddAssign(e)), Ok(_)) => Err(vec![Error::ExpectedIntExpr(*e.span())]),
            (Ok(SubAssign(e)), Ok((&ty, loc))) if ty == Type::Int && e.ty() == Type::Int => {
                Ok(Self::new(loc, SubAssign(e), assign.span))
            }
            (Ok(SubAssign(_)), Ok((&ty, loc))) if ty != Type::Int => {
                Err(vec![Error::ExpectedIntExpr(*loc.span())])
            }
            (Ok(SubAssign(e)), Ok(_)) => Err(vec![Error::ExpectedIntExpr(*e.span())]),
            (Ok(Assign(e)), Ok((&ty, loc))) if ty == e.ty() => {
                Ok(Self::new(loc, Assign(e), assign.span))
            }
            (Ok(Assign(e)), Ok((&ty, loc))) => Err(vec![Error::TypeMismatch {
                rspan: *e.span(),
                rhs: e.ty(),
                lspan: *loc.span(),
                lhs: ty,
            }]),
            (Ok(_), Err(e)) | (Err(e), Ok(_)) => Err(e),
            (Err(mut e1), Err(e2)) => {
                e1.extend(e2);
                Err(e1)
            }
        }
    }
}

fn construct_sig_hashmap<'a>(
    externs: &[PImport<'a>],
) -> Result<HashMap<Span<'a>, FunctionSig<'a>>, Vec<Error<'a>>> {
    let mut errors = vec![];
    for i in 0..externs.len() {
        for j in 0..i {
            if externs[i].name() == externs[j].name() {
                errors.push(Error::Redifinition(
                    *externs[i].name().span(),
                    *externs[j].name().span(),
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(externs
            .iter()
            .map(|f| (*f.name().span(), FunctionSig::from_pimport(f)))
            .collect())
    } else {
        Err(errors)
    }
}

fn construct_var_hashmap(
    vars: Vec<Var<Span<'_>>>,
) -> Result<HashMap<Span<'_>, Var<Span<'_>>>, Vec<Error<'_>>> {
    let mut errors = vec![];
    for i in 0..vars.len() {
        for j in 0..i {
            if vars[i].name() == vars[j].name() {
                errors.push(Error::Redifinition(*vars[i].name(), *vars[j].name()));
            }
        }
    }
    errors.extend(vars.iter().filter_map(|v| match v {
        Var::Scalar { .. } => None,
        Var::Array { size, .. } => {
            let size_span = *size.span();
            if let Ok(size) =
                HIRLiteral::from_pliteral((*size).into(), false).map(|size| size.int().unwrap())
            {
                if size == 0 {
                    Some(Error::ZeroArraySize(size_span))
                } else {
                    None
                }
            } else {
                Some(Error::TooLargeInt(size_span))
            }
        }
    }));
    if !errors.is_empty() {
        Err(errors)
    } else {
        vars.into_iter().map(|v| Ok((*v.name(), v))).collect()
    }
}

impl<'a> StmtBlock<'a> {
    fn from_pblock(
        block: PBlock<'a>,
        in_loop: bool,
        expected_return: Option<Type>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        let block_vst = construct_var_hashmap(block.decls)?;
        let stmts = block
            .stmts
            .into_iter()
            .map(|stmt| {
                HIRStmt::from_pstmt(
                    stmt,
                    in_loop,
                    expected_return,
                    VSymMap::new(&block_vst).parent(&vst),
                    fst,
                )
            })
            .fold_result()?;
        Ok(Self {
            decls: block_vst,
            stmts,
        })
    }
}

impl<'a> HIRStmt<'a> {
    fn from_pstmt(
        stmt: PStmt<'a>,
        in_loop: bool,
        expected_return: Option<Type>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        match stmt {
            ast::Stmt::Call(call) => HIRCall::from_pcall(call, vst, fst).map(Self::Call),
            ast::Stmt::Return { span, expr } => match expr {
                Some(expr) => HIRExpr::from_pexpr(expr, vst, fst)
                    .and_then(|res| match expected_return {
                        None => Err(vec![Error::ReturnValueFromVoid(span)]),
                        Some(Type::Int) if res.ty() == Type::Int => Ok(res),
                        Some(Type::Int) => Err(vec![Error::ExpectedIntExpr(*res.span())]),
                        Some(Type::Bool) if res.ty() == Type::Bool => Ok(res),
                        Some(Type::Bool) => Err(vec![Error::ExpectedBoolExpr(*res.span())]),
                    })
                    .map(|expr| Self::Return {
                        span,
                        expr: Some(expr),
                    }),
                None => {
                    if expected_return.is_some() {
                        Err(vec![Error::ExpectedExpression(span)])
                    } else {
                        Ok(Self::Return { span, expr: None })
                    }
                }
            },
            ast::Stmt::Break(span) => {
                if in_loop {
                    Ok(Self::Break(span))
                } else {
                    Err(vec![Error::BreakOutsideLoop(span)])
                }
            }
            ast::Stmt::Continue(span) => {
                if in_loop {
                    Ok(Self::Continue(span))
                } else {
                    Err(vec![Error::ContinueOutsideLoop(span)])
                }
            }
            ast::Stmt::Assign(assign) => {
                HIRAssign::from_passign(assign, vst, fst).map(Self::Assign)
            }
            ast::Stmt::If {
                cond,
                yes,
                no,
                span,
            } => {
                let cond = HIRExpr::from_pexpr(cond, vst, fst);
                let yes = HIRBlock::from_pblock(yes, in_loop, expected_return, vst, fst);
                let no = no.map(|no| HIRBlock::from_pblock(no, in_loop, expected_return, vst, fst));
                if cond.is_ok() && yes.is_ok() && no.is_none() {
                    if cond.as_ref().unwrap().ty() != Type::Bool {
                        Err(vec![Error::ExpectedBoolExpr(
                            *cond.as_ref().unwrap().span(),
                        )])
                    } else {
                        Ok(Self::If {
                            cond: cond.unwrap(),
                            yes: yes.unwrap(),
                            no: None,
                            span,
                        })
                    }
                } else if cond.is_ok()
                    && yes.is_ok()
                    && no.is_some()
                    && no.as_ref().unwrap().is_ok()
                {
                    if cond.as_ref().unwrap().ty() != Type::Bool {
                        Err(vec![Error::ExpectedBoolExpr(
                            *cond.as_ref().unwrap().span(),
                        )])
                    } else {
                        Ok(Self::If {
                            cond: cond.unwrap(),
                            yes: yes.unwrap(),
                            no: no.unwrap().ok(),
                            span,
                        })
                    }
                } else {
                    let mut errors = vec![];
                    if let Some(e) = cond.err() {
                        errors.extend(e)
                    }
                    if let Some(e) = yes.err() {
                        errors.extend(e)
                    }
                    if let Some(no) = no {
                        if let Some(e) = no.err() {
                            errors.extend(e)
                        }
                    }
                    Err(errors)
                }
            }
            ast::Stmt::While { cond, body, span } => {
                let cond = HIRExpr::from_pexpr(cond, vst, fst);
                let body = HIRBlock::from_pblock(body, true, expected_return, vst, fst);
                if cond.is_ok() && body.is_ok() {
                    if cond.as_ref().unwrap().is_boolean() {
                        Ok(Self::While {
                            cond: cond.unwrap(),
                            body: body.unwrap(),
                            span,
                        })
                    } else {
                        Err(vec![Error::ExpectedBoolExpr(*cond.unwrap().span())])
                    }
                } else {
                    let mut errors = vec![];
                    if let Some(e) = cond.err() {
                        errors.extend(e)
                    }
                    if let Some(e) = body.err() {
                        errors.extend(e)
                    }
                    Err(errors)
                }
            }
            ast::Stmt::For {
                init,
                cond,
                update,
                body,
                span,
            } => {
                let init = HIRAssign::from_passign(init, vst, fst);
                let cond = HIRExpr::from_pexpr(cond, vst, fst);
                let update = HIRAssign::from_passign(update, vst, fst);
                let body = StmtBlock::from_pblock(body, true, expected_return, vst, fst);
                if init.is_ok() && cond.is_ok() && update.is_ok() && body.is_ok() {
                    if cond.as_ref().unwrap().is_boolean() {
                        Ok(Self::For {
                            init: init.unwrap(),
                            cond: cond.unwrap(),
                            update: update.unwrap(),
                            body: body.unwrap(),
                            span,
                        })
                    } else {
                        Err(vec![Error::ExpectedBoolExpr(
                            *cond.as_ref().unwrap().span(),
                        )])
                    }
                } else {
                    let mut errors = vec![];
                    if let Some(e) = init.err() {
                        errors.extend(e)
                    }
                    if let Some(e) = cond.err() {
                        errors.extend(e)
                    }
                    if let Some(e) = update.err() {
                        errors.extend(e)
                    }
                    if let Some(e) = body.err() {
                        errors.extend(e)
                    }
                    Err(errors)
                }
            }
        }
    }
}

fn intersect<'a, O1, O2>(
    map1: &HashMap<Span<'a>, O1>,
    map2: &HashMap<Span<'a>, O2>,
) -> Vec<Error<'a>> {
    map2.iter()
        .filter_map(|(span, _)| map1.get(span).map(|_| Error::Redifinition(*span, *span)))
        .collect()
}

impl<'a> HIRFunction<'a> {
    fn from_pfunction(
        func: PFunction<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        let span = *func.span();
        let args = construct_var_hashmap(func.args)?;
        let body = HIRBlock::from_pblock(
            func.body,
            false,
            func.ret,
            VSymMap::new(&args).parent(&vst),
            fst,
        )?;
        let redefs = intersect(&body.decls, &args);
        if redefs.is_empty() {
            Ok(Self::new(func.name, body, args, func.ret, span))
        } else {
            Err(redefs)
        }
    }
}

impl<'a> HIRRoot<'a> {
    pub fn from_proot(root: PRoot<'a>) -> Result<Self, Vec<Error>> {
        let globals = construct_var_hashmap(root.decls)?;
        let mut sigs = construct_sig_hashmap(&root.imports)?;
        let imports = root
            .imports
            .into_iter()
            .map(|imp| (*imp.name().span(), imp))
            .collect::<HashMap<_, _>>();
        let functions = root
            .funcs
            .into_iter()
            .map(|f| {
                let func_span = *f.name.span();
                if let Some(func) = sigs.get(&func_span) {
                    return Err(vec![Error::Redifinition(func_span, func.name())]);
                }
                sigs.insert(func_span, FunctionSig::from_pfunction(&f));
                let r = HIRFunction::from_pfunction(f, VSymMap::new(&globals), FSymMap::new(&sigs))
                    .map(|f| (*f.name.span(), f));
                r
            })
            .fold_result()?
            .into_iter()
            .collect::<HashMap<_, _>>();
        {
            let mut redefs = intersect(&imports, &globals);
            redefs.extend(intersect(&imports, &functions));
            redefs.extend(intersect(&globals, &functions));
            if redefs.is_empty() {
                Ok(())
            } else {
                Err(redefs)
            }
        }?;
        if let Some((_, f)) = functions.iter().find(|&(s, _)| s.source() == b"main") {
            if f.ret.is_some() || !f.args.is_empty() {
                Err(vec![Error::InvalidMainSig(*f.span())])
            } else {
                Ok(Self {
                    globals,
                    functions,
                    imports,
                })
            }
        } else {
            Err(vec![Error::RootDoesNotContainMain])
        }
    }
}
