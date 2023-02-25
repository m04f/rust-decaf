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

impl StmtChecker {
    pub fn check<'a, F: FnMut(Error<'a>)>(stmt: &PStmt<'a>, mut callback: F) {
        let mut check_nested_expr = |e: &PExpr<'a>| {
            if let ExprInner::Nested(..) = e.inner {
            } else {
                callback(Error::WrapInParens(*e.span()))
            }
        };
        match stmt {
            Stmt::If { cond, .. } => check_nested_expr(cond),
            Stmt::While { cond, .. } => check_nested_expr(cond),
            Stmt::For { init, update, .. } => {
                if let AssignExpr::Assign(..) = init.op {
                } else {
                    callback(Error::ForInitHasToBeAssign(*init.span()))
                }
                if let AssignExpr::Assign(..) = update.op {
                    callback(Error::ForUpdateIsIncOrCompound(*update.span()))
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
                callback(Error::DeclAfterFunc {
                    decl_pos: elem.span(),
                    hinted_pos: self.decls_next_pos.unwrap(),
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

    pub fn check<F: FnMut(Error<'a>)>(&mut self, elem: &PDocElem<'a>, mut callback: F) {
        match elem {
            DocElem::Import(import) if self.imports_finised => {
                assert!(self.imports_next_pos.is_some());
                let error = if self.decls_finished {
                    Error::ImportAfterDecl {
                        import_pos: *import.span(),
                        hinted_pos: self.imports_next_pos.unwrap(),
                    }
                } else {
                    Error::ImportAfterFunc {
                        import_pos: *import.span(),
                        hinted_pos: self.imports_next_pos.unwrap(),
                    }
                };
                callback(error)
            }
            DocElem::Import(..) => {}
            DocElem::Decl(_, span) if self.decls_finished => {
                assert!(self.decls_next_pos.is_some());
                callback(Error::DeclAfterFunc {
                    decl_pos: *span,
                    hinted_pos: self.decls_next_pos.unwrap(),
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
