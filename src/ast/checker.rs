use super::*;
pub struct AstChecker {}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BlockChecker<S> {
    decls_finished: bool,
    decls_next_pos: Option<S>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StmtChecker;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootChecker<S> {
    imports_finised: bool,
    imports_next_pos: Option<S>,
    decls_finished: bool,
    decls_next_pos: Option<S>,
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
pub enum Error<S> {
    MoveDeclTo { pos: S, ctx: Context },
    Illegal { pos: S, ctx: Context },
    WrapExpr { span: S, ctx: Context },
}

impl StmtChecker {
    pub fn check<S: Clone, F: FnMut(Error<S>)>(stmt: &Stmt<S>, mut callback: F) {
        let check_nested_expr = |e: &Expr<S>, ctx: Context, mut callback: F| {
            if let Expr::Nested(..) = e {
                ()
            } else {
                callback(Error::WrapExpr {
                    span: e.span().clone(),
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
                        pos: init.span().clone(),
                        ctx: Context::For,
                    })
                }
                if let AssignExpr::Assign(..) = update.op {
                    callback(Error::Illegal {
                        pos: update.span().clone(),
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

impl<S: Default + Clone> BlockChecker<S> {
    pub fn new() -> Self {
        BlockChecker {
            decls_finished: false,
            decls_next_pos: None,
        }
    }

    pub fn check<F: FnMut(Error<S>)>(&mut self, elem: &BlockElem<S>, mut callback: F) {
        match elem {
            BlockElem::Decl { .. } if self.decls_finished => {
                assert!(self.decls_next_pos.is_some());
                callback(Error::MoveDeclTo {
                    pos: self.decls_next_pos.clone().unwrap(),
                    ctx: Context::Block,
                })
            }
            BlockElem::Decl { .. } => {}
            BlockElem::Stmt(stmt) => {
                if !self.decls_finished {
                    self.decls_finished = true;
                    self.decls_next_pos = Some(stmt.span().clone());
                }
                self.decls_next_pos = None;
            }
            BlockElem::Func(..) => {
                self.decls_next_pos = None;
            }
        }
    }
}

impl<S: Clone> RootChecker<S> {
    pub fn new() -> Self {
        RootChecker {
            imports_finised: false,
            imports_next_pos: None,
            decls_finished: false,
            decls_next_pos: None,
        }
    }

    pub fn check<F: FnMut(Error<S>)>(&mut self, elem: &DocElem<S>, mut callback: F) {
        match elem {
            DocElem::Import(..) if self.imports_finised => {
                assert!(self.imports_next_pos.is_some());
                callback(Error::MoveDeclTo {
                    pos: self.imports_next_pos.clone().unwrap(),
                    ctx: Context::Root,
                })
            }
            DocElem::Import(..) => {}
            DocElem::Decl { .. } if self.decls_finished => {
                assert!(self.decls_next_pos.is_some());
                callback(Error::MoveDeclTo {
                    pos: self.decls_next_pos.clone().unwrap(),
                    ctx: Context::Root,
                })
            }
            DocElem::Decl(..) => {
                if !self.imports_finised {
                    self.imports_finised = true;
                    self.imports_next_pos = Some(elem.span().clone());
                }
                {}
            }
            DocElem::Function(..) => {
                if !self.imports_finised {
                    self.imports_finised = true;
                    self.imports_next_pos = Some(elem.span().clone());
                }
                if !self.decls_finished {
                    self.decls_finished = true;
                    self.decls_next_pos = Some(elem.span().clone());
                }
            }
        }
    }
}
