use crate::{
    hir::*,
    ir::{
        basicblock::{
            BasicBlock, BasicBlockBuilder as Builder, BasicBlockRef, IRFunction, RegAllocator,
            Terminator,
        },
        cfg::*,
        instruction::*,
    },
};

use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone)]
struct Escapes {
    r#return: BasicBlockRef,
    r#break: Option<BasicBlockRef>,
    r#continue: Option<BasicBlockRef>,
}

impl Escapes {
    fn new(r#return: BasicBlockRef) -> Self {
        Self {
            r#return,
            r#break: None,
            r#continue: None,
        }
    }
    fn with_loop_escapes(self, r#break: BasicBlockRef, r#continue: BasicBlockRef) -> Self {
        Self {
            r#break: Some(r#break),
            r#continue: Some(r#continue),
            ..self
        }
    }
}

impl<'a> HIRExpr<'a> {
    fn destruct(self, reg_allocator: &mut RegAllocator) -> (Cfg, Reg) {
        use HIRExpr::*;
        use HIRLoc::*;
        match self {
            Len(length) => {
                let reg = reg_allocator.alloc();
                (
                    Builder::new()
                        .with_instructions([Instruction::new_load_imm(reg, (length as i64).into())])
                        .build_cfg(),
                    reg,
                )
            }
            Neg(expr) => {
                let (expr_cfg, res_reg) = expr.destruct(reg_allocator);
                let negated_reg = reg_allocator.alloc();
                (
                    expr_cfg.concat(
                        Builder::new()
                            .with_instructions([Instruction::new_neg(negated_reg, res_reg)])
                            .build_cfg(),
                    ),
                    negated_reg,
                )
            }
            Not(expr) => {
                let (expr_cfg, res_reg) = expr.destruct(reg_allocator);
                let negated_reg = reg_allocator.alloc();
                (
                    expr_cfg.concat(
                        Builder::new()
                            .with_instructions([Instruction::new_not(negated_reg, res_reg)])
                            .build_cfg(),
                    ),
                    negated_reg,
                )
            }
            Arith { op, lhs, rhs } => {
                let (lhs_subcfg, lhs_reg) = lhs.destruct(reg_allocator);
                let (rhs_subcfg, rhs_reg) = rhs.destruct(reg_allocator);
                let reg = reg_allocator.alloc();
                (
                    lhs_subcfg.concat(rhs_subcfg).concat(
                        Builder::new()
                            .with_instructions([Instruction::new_arith(reg, lhs_reg, op, rhs_reg)])
                            .build_cfg(),
                    ),
                    reg,
                )
            }
            Eq { op, lhs, rhs } => {
                let (lhs_subcfg, lhs_reg) = lhs.destruct(reg_allocator);
                let (rhs_subcfg, rhs_reg) = rhs.destruct(reg_allocator);
                let reg = reg_allocator.alloc();
                (
                    lhs_subcfg.concat(rhs_subcfg).concat(
                        Builder::new()
                            .with_instructions([Instruction::new_eq(reg, lhs_reg, op, rhs_reg)])
                            .build_cfg(),
                    ),
                    reg,
                )
            }
            Cond { .. } => {
                todo!()
            }
            Rel { op, lhs, rhs } => {
                let (lhs_subcfg, lhs_reg) = lhs.destruct(reg_allocator);
                let (rhs_subcfg, rhs_reg) = rhs.destruct(reg_allocator);
                let res_reg = reg_allocator.alloc();
                (
                    lhs_subcfg.concat(rhs_subcfg).concat(
                        Builder::new()
                            .with_instructions([Instruction::new_rel(
                                res_reg, lhs_reg, op, rhs_reg,
                            )])
                            .build_cfg(),
                    ),
                    res_reg,
                )
            }
            Ter { cond, yes, no } => {
                let (cond_subcfg, cond_reg) = cond.destruct(reg_allocator);
                let (yes_subcfg, yes_reg) = yes.destruct(reg_allocator);
                let (no_subcfg, no_reg) = no.destruct(reg_allocator);
                let res_reg = reg_allocator.alloc();
                (
                    cond_subcfg.concat(yes_subcfg).concat(no_subcfg).concat(
                        Builder::new()
                            .with_instructions([Instruction::new_select(
                                res_reg, cond_reg, yes_reg, no_reg,
                            )])
                            .build_cfg(),
                    ),
                    res_reg,
                )
            }
            Literal(literal) => {
                let reg = reg_allocator.alloc();
                (
                    Builder::new()
                        .with_instructions([Instruction::new_load_imm(reg, literal.into())])
                        .build_cfg(),
                    reg,
                )
            }
            Loc(loc) => match *loc {
                Scalar(var) => {
                    let res_reg = reg_allocator.alloc();
                    (
                        Builder::new()
                            .with_instructions([Instruction::new_load(res_reg, var.into_val())])
                            .build_cfg(),
                        res_reg,
                    )
                }
                Index { arr, index, .. } => {
                    let (index_subcfg, index_reg) = index.destruct(reg_allocator);
                    let res_reg = reg_allocator.alloc();
                    (
                        index_subcfg.concat(
                            Builder::new()
                                .with_instructions([Instruction::new_load_offset(
                                    res_reg,
                                    arr.into_val(),
                                    index_reg,
                                )])
                                .build_cfg(),
                        ),
                        res_reg,
                    )
                }
            },
            Call(call) => {
                let (cfg, res) = call.destruct(reg_allocator);
                (cfg, res.unwrap())
            }
        }
    }
}

impl<'a> HIRStmt<'a> {
    fn destruct(self, reg_allocator: &mut RegAllocator, escapes: Escapes) -> Cfg {
        use HIRStmt::*;
        match self {
            Expr(expr) => expr.destruct(reg_allocator).0,
            Break => escapes.r#break.unwrap().into(),
            Continue => escapes.r#continue.unwrap().into(),
            Return(expr) => expr.map_or_else(
                || escapes.r#return.into(),
                |expr| {
                    let (subcfg, expr_reg) = expr.destruct(reg_allocator);
                    subcfg.concat(
                        Builder::new()
                            .with_instructions([Instruction::new_return(expr_reg)])
                            .build_cfg(),
                    )
                },
            ),
            Assign(assign) => assign.destruct(reg_allocator),
            If { cond, yes, no } => {
                let (cond_subcfg, cond_reg) = cond.destruct(reg_allocator);
                let join = Builder::new().build_cfg();
                let yes_subcfg = yes
                    .destruct(reg_allocator, escapes.clone())
                    .concat(join.clone());
                let no_subcfg = no.destruct(reg_allocator, escapes).concat(join);
                cond_subcfg.concat(
                    Builder::new()
                        .with_fork(cond_reg, yes_subcfg.beg, no_subcfg.beg)
                        .build_cfg(),
                )
            }
            While { .. } => todo!(),
            For { .. } => todo!(),
        }
    }
}

impl<'a> HIRBlock<'a> {
    fn destruct(self, reg_allocator: &mut RegAllocator, escapes: Escapes) -> Cfg {
        let decls = Builder::new()
            .with_instructions(self.decls.into_keys().map(Instruction::new_stack_alloc))
            .build_cfg();
        self.stmts
            .into_iter()
            .map(|stmt| stmt.destruct(reg_allocator, escapes.clone()))
            .fold(decls, |cfg, tail| cfg.concat(tail))
    }
}

impl<'a> HIRAssign<'a> {
    fn destruct(self, reg_allocator: &mut RegAllocator) -> Cfg {
        use HIRLoc::*;
        let HIRAssign { lhs, rhs } = self;
        let (rhs_subcfg, rhs_reg) = rhs.destruct(reg_allocator);
        match lhs {
            Scalar(var) => {
                let rhs_reg = reg_allocator.alloc();
                rhs_subcfg.concat(
                    Builder::new()
                        .with_instructions([Instruction::new_store(var.into_val(), rhs_reg)])
                        .build_cfg(),
                )
            }
            Index { arr, index, .. } => {
                let (index_subcfg, index_reg) = index.destruct(reg_allocator);
                rhs_subcfg.concat(index_subcfg).concat(
                    Builder::new()
                        .with_instructions([Instruction::new_store_offset(
                            arr.into_val(),
                            index_reg,
                            rhs_reg,
                        )])
                        .build_cfg(),
                )
            }
        }
    }
}

impl<'a> HIRCall<'a> {
    fn destruct(self, reg_allocator: &mut RegAllocator) -> (Cfg, Option<Reg>) {
        use HIRCall::*;
        match self {
            Extern { .. } => todo!(),
            Decaf { name, ret, args } => {
                let (args_cfg, args) = args
                    .into_iter()
                    .map(|arg| arg.destruct(reg_allocator))
                    .fold(
                        (Builder::new().build_cfg(), vec![]),
                        |(cfg, mut args_regs), (tail_cfg, new_arg)| {
                            (cfg.concat(tail_cfg), {
                                args_regs.push(new_arg);
                                args_regs
                            })
                        },
                    );
                if ret.is_some() {
                    let ret_reg = reg_allocator.alloc();
                    (
                        args_cfg.concat(
                            Builder::new()
                                .with_instructions([Instruction::new_ret_call(ret_reg, name, args)])
                                .build_cfg(),
                        ),
                        Some(ret_reg),
                    )
                } else {
                    (
                        args_cfg.concat(
                            Builder::new()
                                .with_instructions([Instruction::new_void_call(name, args)])
                                .build_cfg(),
                        ),
                        None,
                    )
                }
            }
        }
    }
}

impl<'a> HIRFunction<'a> {
    pub fn destruct(self) -> IRFunction {
        let mut reg_allocator = RegAllocator::new();
        let Cfg { beg: root, end } = if self.ret.is_some() {
            let return_escape = Builder::new()
                .with_instructions([Instruction::ReturnGuard])
                .build_refcount();
            self.body
                .destruct(&mut reg_allocator, Escapes::new(return_escape))
        } else {
            self.body.destruct(
                &mut reg_allocator,
                Escapes::new(Builder::new().build_refcount()),
            )
        };
        drop(end);
        let func = IRFunction::new(root);
        func.root().dfs().for_each(|node: Rc<RefCell<BasicBlock>>| {
            let temp_borrow = node.borrow();
            if let Some((new_terminator, new_instructions)) = match temp_borrow.terminator() {
                Some(Terminator::Tail { block }) if Rc::strong_count(block) == 1 => {
                    let mut block_mut = block.borrow_mut();
                    Some((block_mut.take_terminator(), block_mut.take_instructions()))
                }
                _ => None,
            } {
                drop(temp_borrow);
                node.borrow_mut().extend_instructions(new_instructions);
                *node.borrow_mut().terminator_mut() = new_terminator;
            };
        });
        func
    }
}
