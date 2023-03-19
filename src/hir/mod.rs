use crate::{span::Span, parser::{self, ast::*} };

use std::{collections::{HashMap, HashSet}, num::NonZeroU32};

mod ast;
mod error;
use error::*;
use Error::*;
mod sym_map;
use sym_map::*;

pub use sym_map::{FSymMap, VSymMap};
pub use ast::*;

impl<'a> HIRExpr<'a> {
    fn from_pexpr(
        expr: PExpr<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        use PExpr::*;
        match expr {
            Len(_, ident) => match vst.get_sym(ident.span()) {
                None => Err(vec![UndeclaredIdentifier(ident.span())]),
                Some(var) => match var {
                    HIRVar::Scalar { .. } => Err(vec![ExpectedArrayVariable(ident.span())]),
                    HIRVar::Array { size, .. } => Ok(Self::Len(*size)),
                },
            },
            Not(span, e) => {
                let e = Self::from_pexpr(*e, vst, fst)?;
                if !e.is_boolean() {
                    Err(vec![ExpectedBoolExpr(span)])
                } else {
                    Ok(e.new_not())
                }
            }
            Neg(_, e) if let Literal { value, .. } = *e => {
                HIRLiteral::from_pliteral(value, true).map(|value| value.into())
            }
            Neg(span, e) => {
                let e = Self::from_pexpr(*e, vst, fst)?;
                if !e.is_int() {
                    Err(vec![ExpectedIntExpr(span)])
                } else {
                    Ok(Self::new_neg(e))
                }
            }
            Nested(_, e) => {
                let e = Self::from_pexpr(*e, vst, fst)?;
                Ok(e.nested())
            }
            Index { name, offset, span } => match vst.get_sym(name.span()) {
                None => Err(vec![UndeclaredIdentifier(name.span())]),
                Some(HIRVar::Scalar { .. }) => {
                    Err(vec![ExpectedArrayVariable(name.span())])
                }
                Some(var) => {
                    let offset = Self::from_pexpr(*offset, vst, fst)?;
                    if !offset.is_int() {
                        Err(vec![ExpectedIntExpr(span)])
                    } else {
                        Ok(HIRLoc::index(*var, offset).into())
                    }
                }
            },
            Scalar(ident) => match vst.get_sym(ident.span()) {
                None => Err(vec![UndeclaredIdentifier(ident.span())]),
                Some(HIRVar::Scalar(var)) => Ok(HIRLoc::Scalar(*var).into()),
                _ => Err(vec![ExpectedScalarVariable(ident.span())]),
            },
            Ter {
                cond,
                yes,
                no,
                ..
            } => {
                let cond_span = cond.span();
                let yes_span = yes.span();
                let no_span = no.span();
                let cond = Self::from_pexpr(*cond, vst, fst);
                let yes = Self::from_pexpr(*yes, vst, fst);
                let no = Self::from_pexpr(*no, vst, fst);
                match (cond, yes, no) {
                    (Ok(cond), Ok(yes), Ok(no)) => {
                        if cond.is_boolean() && yes.r#type() == no.r#type() {
                            Ok(Self::Ter { cond: Box::new(cond), yes: Box::new(yes), no: Box::new(no) })
                        } else {
                            let mut errors = vec![];
                            (!cond.is_boolean())
                                .then(|| errors.push(ExpectedBoolExpr(cond_span)));
                            (yes.r#type() != no.r#type()).then(|| {
                                errors.push(TypeMismatch {
                                    rspan: yes_span,
                                    rhs: yes.r#type(),
                                    lspan: no_span,
                                    lhs: no.r#type(),
                                })
                            });
                            Err(errors)
                        }
                    }
                    (cond, yes, no) => {
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
            }
            BinOp { op, lhs, rhs, .. } => {
                let lspan = lhs.span();
                let rspan = rhs.span();
                let lhs = Self::from_pexpr(*lhs, vst, fst);
                let rhs = Self::from_pexpr(*rhs, vst, fst);
                match (lhs, rhs) {
                    (Ok(lhs), Ok(rhs)) => {
                        if let Ok(op) = EqOp::try_from(op) {
                            if lhs.r#type() == rhs.r#type() {
                                Ok(Self::Eq { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
                            } else {
                                Err(vec![TypeMismatch {
                                    lspan,
                                    lhs: lhs.r#type(),
                                    rspan,
                                    rhs: rhs.r#type(),
                                }])
                            }
                        } else if let Ok(op) = CondOp::try_from(op) {
                            if lhs.r#type() == Type::Bool && rhs.r#type() == Type::Bool {
                                Ok(Self::Cond { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
                            } else {
                                let mut errors = vec![];
                                if lhs.r#type() != Type::Bool {
                                    errors.push(ExpectedBoolExpr(lspan))
                                }
                                if rhs.r#type() != Type::Bool {
                                    errors.push(ExpectedBoolExpr(rspan))
                                }
                                Err(errors)
                            }
                        } else if lhs.r#type() == Type::Int && rhs.r#type() == Type::Int {
                            if let Ok(op) = ArithOp::try_from(op) {
                                Ok(Self::Arith { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
                            } else if let Ok(op) = RelOp::try_from(op) {
                                Ok(Self::Rel { op, lhs: Box::new(lhs), rhs: Box::new(rhs) })
                            } else {
                                unreachable!()
                            }
                        } else {
                            let mut errors = vec![];
                            if lhs.r#type() != Type::Int {
                                errors.push(ExpectedIntExpr(lspan))
                            }
                            if rhs.r#type() != Type::Int {
                                errors.push(ExpectedIntExpr(rspan))
                            }
                            Err(errors)
                        }
                    }
                    (lhs, rhs) => {
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
            }
            Literal { value, .. } => {
                HIRLiteral::from_pliteral(value, false).map(|value| value.into())
            }
            Call(call) => {
                let call_span = call.span();
                HIRCall::from_pcall(call, vst, fst).and_then(|call| {
                    if let HIRCall::Decaf { ret: None, .. } = call {
                        Err(vec![VoidFuncAsExpr(call_span)])
                    } else {
                        Ok(call)
                    }
                })
                    .map(HIRExpr::Call)
            }
        }
    }
}

impl<'a> HIRLiteral {
    fn from_pliteral(literal: PLiteral<'a>, is_neg: bool) -> Result<Self, Vec<Error<'a>>> {
        let map_digit = if is_neg { |dig: i64| -dig } else { |dig| dig };
        match literal {
            PLiteral::Decimal(num) => {
                let parsed = num.bytes().try_fold(0, |acc: i64, digit| {
                    acc.checked_mul(10)
                        .and_then(|acc| acc.checked_add(map_digit((digit - b'0') as i64)))
                });
                parsed
                    .ok_or(vec![TooLargeInt(num)])
                    .map(HIRLiteral::Int)
            }
            PLiteral::Hex(num) => num
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
                .ok_or(vec![TooLargeInt(num)])
                .map(HIRLiteral::Int),
            PLiteral::Bool(val) => Ok(HIRLiteral::Bool(val)),
            PLiteral::Char(c) => Ok(HIRLiteral::Int(c.into())),
        }
    }
}

impl<'a> ExternArg<'a> {
    fn extern_from_pcall(
        arg: PArg<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        match arg {
            PArg::Expr(PExpr::Scalar(ident)) 
                if let Some(ident) = 
                              vst.get_sym(ident.span()).and_then(|var| var.is_array().then_some(var.name()))
                    => {
                        Ok(Self::Array(ident))
                    }
            PArg::Expr(e) => HIRExpr::from_pexpr(e, vst, fst).map(Self::Expr),
            PArg::String(s) => Ok(s.into()),
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
        if vst.get_sym(call.name.span()).is_some() {
            Err(vec![VariableNotAMethod(call.name.span())])
        } else {
            match fst.get_sym(call.name.span()) {
                None => Err(vec![UndeclaredIdentifier(call.name.span())]),
                Some(FunctionSig::Extern(name)) => call
                    .args
                    .into_iter()
                    .map(|arg| ExternArg::extern_from_pcall(arg, vst, fst))
                    .fold_result()
                    .map(|args| HIRCall::new_extern(*name, args) ),
                Some(FunctionSig::Decl { name, arg_types, ty }) => {
                    if call.args.len() == arg_types.len() {
                        call.args
                            .into_iter()
                            .zip(arg_types.iter())
                            .map(|(arg, r#type)| {
                                match arg {
                                    PArg::String(s) => Err(vec![StringInUserDefined(s.span())]),
                                    PArg::Expr(expr) => {
                                    let span = expr.span();
                                    HIRExpr::from_pexpr(expr, vst, fst).and_then(|expr| {
                                    if expr.r#type() != *r#type {
                                        Err(vec![ExpectedType {
                                            expected: *r#type,
                                            span,
                                            found: expr.r#type(),
                                        }])
                                    } else {
                                        Ok(expr)
                                    }
                                })}
                                }
                            })
                            .fold_result()
                            .map(|args| HIRCall::new_decaf(*name, *ty, args))
                    } else {
                        Err(vec![WrongNumberOfArgs {
                            expected: arg_types.len(),
                            found: call.args.len(),
                            span: call.name.span(),
                        }])
                    }
                }
            }
        }
    }
}

impl<'a> HIRLoc<'a> {
    fn from_ploc(loc: PLoc<'a>, vst: VSymMap<'a, '_>, fst: FSymMap<'a, '_>) -> Result<Self, Vec<Error<'a>>> {
        match (vst.get_sym(loc.ident()), loc.offset) {
            (Some(HIRVar::Scalar(var)), None) => Ok(HIRLoc::Scalar(*var)),
            (Some(HIRVar::Scalar(var)), Some(_)) => Err(vec![CannotIndexScalar(var.into_val())]),
            (Some(HIRVar::Array{ arr, .. }), None) => Err(vec![CannotAssignToArray(arr.into_val())]),
            (Some(arr), Some(offset)) => {
                let offset_span = offset.span();
                let offset = HIRExpr::from_pexpr(offset, vst, fst)?;
                if offset.r#type() == Type::Int {
                    Ok(HIRLoc::index(*arr, offset))
                } else {
                    Err(vec![ExpectedIntExpr(offset_span)])
                }
            }
            ( None, _ ) => Err(vec![UndeclaredIdentifier(loc.ident.span())]),
        }
    }
}

impl<'a> HIRAssign<'a> {
    fn from_passign(
        assign: PAssign<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        use parser::ast::PAssignExpr::*;
        use ArithOp::*;
        let lhs_span = assign.lhs.span();
        let loc = HIRLoc::from_ploc(assign.lhs, vst, fst)?;
        match assign.op {
            Inc => {
                if loc.r#type() != Type::Int {
                    Err(vec![IncNonInt(lhs_span)])
                } else {
                    Ok(Self { lhs: loc.clone(), rhs: HIRExpr::Arith { op: Add, lhs: Box::new(loc.into()), rhs: Box::new(HIRLiteral::from(1).into()) }})
                }
            },
            Dec => {
                if loc.r#type() != Type::Int {
                    Err(vec![DecNonInt(lhs_span)])
                } else {
                    Ok(Self { lhs: loc.clone(), rhs: HIRExpr::Arith { op: Sub, lhs: Box::new(loc.into()), rhs: Box::new(HIRLiteral::from(1).into()) }})
                }
            },
            Assign(expr) => {
                let rhs = HIRExpr::from_pexpr(expr, vst, fst)?;
                if rhs.r#type() != loc.r#type() {
                    Err(vec![AssignOfDifferentType{ lhs: lhs_span, ltype: loc.r#type(), rtype: rhs.r#type() }])
                } else {
                    Ok(Self { lhs: loc, rhs })
                }
            }
            AddAssign(expr) => {
                let rhs_span = expr.span();
                let rhs = HIRExpr::from_pexpr(expr, vst, fst)?;
                if rhs.r#type() != Type::Int {
                    Err(vec![ExpectedIntExpr(rhs_span)])
                } else if loc.r#type() != Type::Int {
                    Err(vec![ExpectedIntExpr(lhs_span)])
                } else {
                    Ok(Self { lhs: loc.clone(), rhs: HIRExpr::Arith { op: Add, lhs: Box::new(loc.into()), rhs: Box::new(rhs) }})
                }
            }
            SubAssign(expr) => {
                let rhs_span = expr.span();
                let rhs = HIRExpr::from_pexpr(expr, vst, fst)?;
                if rhs.r#type() != Type::Int {
                    Err(vec![ExpectedIntExpr(rhs_span)])
                } else if loc.r#type() != Type::Int {
                    Err(vec![ExpectedIntExpr(lhs_span)])
                } else {
                    Ok(Self { lhs: loc.clone(), rhs: HIRExpr::Arith { op: Sub, lhs: Box::new(loc.into()), rhs: Box::new(rhs) }})
                }
            }
        }

    }
}

impl<'a> HIRVar<'a> {
fn from_pvar(
        var: PVar<'a>,
    ) -> Result<Self, Error<'a>> {
        match var {
            PVar::Scalar{ ident, ty } => {
                Ok(HIRVar::Scalar(Typed::new(ty, ident.span())))
            }
            PVar::Array { ident, size, ty, .. } => {
                let size_span = size.span();
                if let Ok(size) = HIRLiteral::from_pliteral(PLiteral::from(size), false)
                    .map(|size| size.int().unwrap())
                {
                    NonZeroU32::new(size as u32).map(|size| {
                        Self::Array { arr: Typed::new(ty, ident.span()), size }
                    }).ok_or(ZeroArraySize(size_span))
                } else {
                    Err(TooLargeInt(size_span))
                }
            }
        }
    }
}

impl<'a> HIRBlock<'a> {
    fn from_pblock(
        block: PBlock<'a>,
        in_loop: bool,
        expected_return: Option<Type>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        let block_vst = construct_var_hashmap(block.decls())?;
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
            PStmt::Call(call) => HIRCall::from_pcall(call, vst, fst).map(|call| Self::Expr(call.into())),
            PStmt::Return { span, expr } => match expr {
                Some(expr) => { let expr_span = expr.span(); HIRExpr::from_pexpr(expr, vst, fst)
                    .and_then(|res| match expected_return {
                        None => Err(vec![ReturnValueFromVoid(span)]),
                        Some(Type::Int) if res.r#type() == Type::Int => Ok(res),
                        Some(Type::Int) => Err(vec![ExpectedIntExpr(expr_span)]),
                        Some(Type::Bool) if res.r#type() == Type::Bool => Ok(res),
                        Some(Type::Bool) => Err(vec![ExpectedBoolExpr(expr_span)]),
                    }) }
                    .map(|expr| Self::Return ( Some(expr),)),
                None => {
                    if expected_return.is_some() {
                        Err(vec![ExpectedExpression(span)])
                    } else {
                        Ok(Self::Return (None))
                    }
                }
            },
            PStmt::Break(span) => {
                if in_loop {
                    Ok(Self::Break)
                } else {
                    Err(vec![BreakOutsideLoop(span)])
                }
            }
            PStmt::Continue(span) => {
                if in_loop {
                    Ok(Self::Continue)
                } else {
                    Err(vec![ContinueOutsideLoop(span)])
                }
            }
            PStmt::Assign(assign) => {
                HIRAssign::from_passign(assign, vst, fst).map(Self::Assign)
            }
            PStmt::If {
                cond,
                yes,
                no,
                ..
            } => {
                let cond_span = cond.span();
                let cond = HIRExpr::from_pexpr(cond, vst, fst);
                let yes = HIRBlock::from_pblock(yes, in_loop, expected_return, vst, fst);
                let no = no.map(|no| HIRBlock::from_pblock(no, in_loop, expected_return, vst, fst));
                match (cond, yes, no) {
                    (Ok(cond), Ok(yes), None) => {
                        if cond.r#type() != Type::Bool {
                            Err(vec![ExpectedBoolExpr(cond_span)])
                        } else {
                            Ok(Self::If {
                                cond,
                                yes: Box::new(yes),
                                no: Box::default()
                            })
                        }
                    }
                    (Ok(cond), Ok(yes), Some(Ok(no))) => {
                        if cond.r#type() != Type::Bool {
                            Err(vec![ExpectedBoolExpr(cond_span)])
                        } else {
                            Ok(Self::If {
                                cond,
                                yes: Box::new(yes),
                                no: Box::new(no),
                            })
                        }
                    }
                    (cond, yes, no) => {
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
            }
            PStmt::While { cond, body, .. } => {
                let cond_span = cond.span();
                let cond = HIRExpr::from_pexpr(cond, vst, fst);
                let body = HIRBlock::from_pblock(body, true, expected_return, vst, fst);
                match (cond, body) {
                    (Ok(cond), Ok(body)) => {
                        if cond.is_boolean() {
                            Ok(Self::While { cond, body: Box::new(body) })
                        } else {
                            Err(vec![ExpectedBoolExpr(cond_span)])
                        }
                    }
                    (cond, body) => {
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
            }
            PStmt::For {
                init,
                cond,
                update,
                body,
                ..
            } => {
                let cond_span = cond.span();
                let init = HIRAssign::from_passign(init, vst, fst);
                let cond = HIRExpr::from_pexpr(cond, vst, fst);
                let update = HIRAssign::from_passign(update, vst, fst);
                let body = HIRBlock::from_pblock(body, true, expected_return, vst, fst);
                match (init, cond, update, body) {
                    (Ok(init), Ok(cond), Ok(update), Ok(body)) => {
                        if cond.is_boolean() {
                            Ok(Self::For {
                                init,
                                cond,
                                update,
                                body: Box::new(body),
                            })
                        } else {
                            Err(vec![ExpectedBoolExpr(cond_span)])
                        }
                    }
                    (init, cond, update, body) => {
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
}

impl<'a> HIRFunction<'a> {
    pub fn from_pfunction(
        func: PFunction<'a>,
        vst: VSymMap<'a, '_>,
        fst: FSymMap<'a, '_>,
    ) -> Result<Self, Vec<Error<'a>>> {
        let args_sorted = func.args.iter().map(|arg| arg.name()).collect();
        let args = construct_var_hashmap(func.args)?;
        let body = HIRBlock::from_pblock(
            func.body,
            false,
            func.ret,
            VSymMap::new(&args).parent(&vst),
            fst,
        )?;
        let redefs = intersect(body.decls().keys().copied(), &args);
        if redefs.is_empty() {
            Ok(Self::new(func.name, body, args, args_sorted, func.ret))
        } else {
            Err(redefs)
        }
    }
}

impl<'a> HIRRoot<'a> {
    pub fn from_proot(root: PRoot<'a>, ensure_main: bool) -> Result<Self, Vec<Error>> {
        let globals = construct_var_hashmap(root.decls)?;
        let mut sigs = construct_sig_hashmap(&root.imports)?;
        let imports = root
            .imports
            .into_iter()
            .map(|imp| imp.name().span())
            .collect::<HashSet<Span>>();
        let functions = root
            .funcs
            .into_iter()
            .map(|f| {
                let func_span = f.name.span();
                if let Some(func) = sigs.get(&func_span) {
                    return Err(vec![Redifinition(func_span, func.name())]);
                }
                sigs.insert(func_span, FunctionSig::get(&f));
                let r = HIRFunction::from_pfunction(f, VSymMap::new(&globals), FSymMap::new(&sigs))
                    .map(|f| (f.name, f));
                r
            })
            .fold_result()?
            .into_iter()
            .collect::<HashMap<_, _>>();
        {
            let mut redefs = intersect(imports.iter().copied(), &globals);
            redefs.extend(intersect(imports.iter().copied(), &functions));
            redefs.extend(intersect(globals.keys().copied(), &functions));
            if redefs.is_empty() {
                Ok(())
            } else {
                Err(redefs)
            }
        }?;
        if let Some((s, f)) = functions.iter().find(|&(s, _)| s.source() == b"main") {
            if f.ret.is_some() || !f.args.is_empty() {
                Err(vec![InvalidMainSig(*s)])
            } else {
                Ok(Self {
                    globals,
                    functions,
                    imports,
                })
            }
        } else if ensure_main {
            Err(vec![RootDoesNotContainMain])
        } else {
            Ok(Self {
                globals,
                functions,
                imports,
            })
        }
    }
}
