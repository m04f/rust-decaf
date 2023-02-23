use super::*;
use crate::parser::*;

pub struct AstChecker {}

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

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Context {
    If,
    While,
    For,
    Block,
    Root,
}

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Error<'a> {
    MoveDeclTo { pos: Span<'a>, ctx: Context },
    Illegal { pos: Span<'a>, ctx: Context },
    WrapExpr { span: Span<'a>, ctx: Context },
}

impl StmtChecker {
    pub fn check<'a, F: FnMut(Error<'a>)>(stmt: &PStmt<'a>, mut callback: F) {
        let check_nested_expr =
            |e: &PExpr<'a>, ctx: Context, mut callback: F| {
                if let ExprInner::Nested(..) = e.inner {
                } else {
                    callback(Error::WrapExpr {
                        span: *e.span(),
                        ctx,
                    })
                }
            };
        match stmt {
            Stmt::If { cond, .. } => check_nested_expr(cond, Context::If, callback),
            Stmt::While { cond, .. } => check_nested_expr(cond, Context::While, callback),
            Stmt::For { init, update, .. } => {
                if let AssignExpr::Assign(..) = init.op {
                } else {
                    callback(Error::Illegal {
                        pos: *init.span(),
                        ctx: Context::For,
                    })
                }
                if let AssignExpr::Assign(..) = update.op {
                    callback(Error::Illegal {
                        pos: *update.span(),
                        ctx: Context::For,
                    })
                } else {
                }
            }
            Stmt::Return { .. }
            | Stmt::Continue(..)
            | Stmt::Break(..)
            | Stmt::Call(..)
            | Stmt::Assign(..) => {}
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
            BlockElem::Decl { .. } if self.decls_finished => {
                assert!(self.decls_next_pos.is_some());
                callback(Error::MoveDeclTo {
                    pos: self.decls_next_pos.unwrap(),
                    ctx: Context::Block,
                })
            }
            BlockElem::Decl { .. } => {}
            BlockElem::Stmt(stmt) => {
                if !self.decls_finished {
                    self.decls_finished = true;
                    self.decls_next_pos = Some(*stmt.span());
                }
            }
            BlockElem::Func(..) => {}
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

    pub fn check<F: FnMut(Error)>(&mut self, elem: &PDocElem<'a>, mut callback: F) {
        match elem {
            DocElem::Import(..) if self.imports_finised => {
                assert!(self.imports_next_pos.is_some());
                callback(Error::MoveDeclTo {
                    pos: self.imports_next_pos.unwrap(),
                    ctx: Context::Root,
                })
            }
            DocElem::Import(..) => {}
            DocElem::Decl { .. } if self.decls_finished => {
                assert!(self.decls_next_pos.is_some());
                callback(Error::MoveDeclTo {
                    pos: self.decls_next_pos.unwrap(),
                    ctx: Context::Root,
                })
            }
            DocElem::Decl(..) => {
                if !self.imports_finised {
                    self.imports_finised = true;
                    self.imports_next_pos = Some(*elem.span());
                }
                {}
            }
            DocElem::Function(..) => {
                if !self.imports_finised {
                    self.imports_finised = true;
                    self.imports_next_pos = Some(*elem.span());
                }
                if !self.decls_finished {
                    self.decls_finished = true;
                    self.decls_next_pos = Some(*elem.span());
                }
            }
        }
    }
}
