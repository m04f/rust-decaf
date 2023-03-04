use crate::{
    hir::*,
    ir::{arena::*, cfg::*, instruction::*},
};

#[derive(Debug, Clone, Copy)]
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

impl SubCfg {
    fn new_if((cond, cond_reg): (Self, Reg), yes: Self, no: Self, arena: &mut Arena) -> Self {
        let join = arena.alloc_block();
        let yes = yes.concat(join.into(), arena);
        let no = no.concat(join.into(), arena);
        arena
            .get_block_mut(cond.end)
            .set_fork(cond_reg, yes.beg, no.beg, arena);
        SubCfg::new(cond.beg, join)
    }
    fn new_loop(
        (cond, cond_reg): (Self, Reg),
        body: HIRBlock<'_>,
        arena: &mut Arena,
        escapes: Escapes,
    ) -> Self {
        let loop_end = arena.alloc_block();
        let body = body.destruct(arena, escapes.with_loop_escapes(loop_end, cond.beg));
        arena
            .get_block_mut(cond.end)
            .set_fork(cond_reg, body.beg, loop_end, arena);
        arena.get_block_mut(body.end).set_tail(cond.beg, arena);
        Self::new(cond.beg, loop_end)
    }
}

impl<'a> HIRExpr<'a> {
    fn destruct(self, arena: &mut Arena) -> (SubCfg, Reg) {
        use HIRExpr::*;
        use HIRLoc::*;
        match self {
            Len(length) => {
                let block_ref = arena.alloc_block();
                let reg = arena.alloc_reg();
                let instruction = Instruction::new_load_imm(reg, (length as i64).into());
                arena.get_block_mut(block_ref).push(instruction);
                (block_ref.into(), reg)
            }
            Neg(expr) => {
                let (subcfg, expr_reg) = expr.destruct(arena);
                let block_ref = arena.alloc_block();
                let reg = arena.alloc_reg();
                arena
                    .get_block_mut(block_ref)
                    .push(Instruction::new_neg(reg, expr_reg));
                (subcfg.concat(block_ref.into(), arena), reg)
            }
            Not(expr) => {
                let (subcfg, expr_reg) = expr.destruct(arena);
                let block_ref = arena.alloc_block();
                let reg = arena.alloc_reg();
                arena
                    .get_block_mut(block_ref)
                    .push(Instruction::new_not(reg, expr_reg));
                (subcfg.concat(block_ref.into(), arena), reg)
            }
            Arith { op, lhs, rhs } => {
                let (lhs_subcfg, lhs_reg) = lhs.destruct(arena);
                let (rhs_subcfg, rhs_reg) = rhs.destruct(arena);
                let block_ref = arena.alloc_block();
                let reg = arena.alloc_reg();
                arena
                    .get_block_mut(block_ref)
                    .push(Instruction::new_arith(reg, lhs_reg, op, rhs_reg));
                (
                    lhs_subcfg
                        .concat(rhs_subcfg, arena)
                        .concat(block_ref.into(), arena),
                    reg,
                )
            }
            Eq { op, lhs, rhs } => {
                let (lhs_subcfg, lhs_reg) = lhs.destruct(arena);
                let (rhs_subcfg, rhs_reg) = rhs.destruct(arena);
                let block_ref = arena.alloc_block();
                let reg = arena.alloc_reg();
                arena
                    .get_block_mut(block_ref)
                    .push(Instruction::new_eq(reg, lhs_reg, op, rhs_reg));
                (
                    lhs_subcfg
                        .concat(rhs_subcfg, arena)
                        .concat(block_ref.into(), arena),
                    reg,
                )
            }
            Cond { op, lhs, rhs } => {
                let (lhs_subcfg, lhs_reg) = lhs.destruct(arena);
                let (rhs_subcfg, rhs_reg) = rhs.destruct(arena);
                let join = arena.alloc_block();
                let result_reg = arena.alloc_reg();
                let set_false = arena.alloc_block();
                let set_false_instruction =
                    Instruction::new_load_imm(result_reg, Immediate::Bool(false));
                let set_true = arena.alloc_block();
                let set_true_instruction =
                    Instruction::new_load_imm(result_reg, Immediate::Bool(true));
                arena.get_block_mut(set_false).push(set_false_instruction);
                arena.get_block_mut(set_true).push(set_true_instruction);
                arena.get_block_mut(set_true).set_tail(join, arena);
                arena.get_block_mut(set_false).set_tail(join, arena);
                match op {
                    CondOp::Or => {
                        arena
                            .get_block_mut(rhs_subcfg.end)
                            .set_fork(rhs_reg, set_true, set_false, arena);
                        arena.get_block_mut(lhs_subcfg.end).set_fork(
                            rhs_reg,
                            set_true,
                            rhs_subcfg.beg,
                            arena,
                        );
                        (SubCfg::new(lhs_subcfg.beg, join), result_reg)
                    }
                    CondOp::And => {
                        let neg_block = arena.alloc_block();
                        let lhs_neg_reg = arena.alloc_reg();
                        let neg_instruction = Instruction::new_not(lhs_neg_reg, lhs_reg);
                        arena
                            .get_block_mut(lhs_subcfg.end)
                            .set_tail(neg_block, arena);
                        arena.get_block_mut(neg_block).push(neg_instruction);
                        arena
                            .get_block_mut(rhs_subcfg.end)
                            .set_fork(rhs_reg, set_true, set_false, arena);
                        arena.get_block_mut(neg_block).set_fork(
                            lhs_neg_reg,
                            set_false,
                            rhs_subcfg.beg,
                            arena,
                        );
                        (SubCfg::new(lhs_subcfg.beg, join), result_reg)
                    }
                }
            }
            Rel { op, lhs, rhs } => {
                let (lhs_subcfg, lhs_reg) = lhs.destruct(arena);
                let (rhs_subcfg, rhs_reg) = rhs.destruct(arena);
                let block_ref = arena.alloc_block();
                let reg = arena.alloc_reg();
                arena
                    .get_block_mut(block_ref)
                    .push(Instruction::new_rel(reg, lhs_reg, op, rhs_reg));
                (
                    lhs_subcfg
                        .concat(rhs_subcfg, arena)
                        .concat(block_ref.into(), arena),
                    reg,
                )
            }
            Ter { cond, yes, no } => {
                let (cond_subcfg, cond_reg) = cond.destruct(arena);
                let (yes_subcfg, yes_reg) = yes.destruct(arena);
                let (no_subcfg, no_reg) = no.destruct(arena);
                let block_ref = arena.alloc_block();
                let reg = arena.alloc_reg();
                let instruction = Instruction::new_select(reg, cond_reg, yes_reg, no_reg);
                arena.get_block_mut(block_ref).push(instruction);
                (
                    cond_subcfg
                        .concat(yes_subcfg, arena)
                        .concat(no_subcfg, arena)
                        .concat(block_ref.into(), arena),
                    reg,
                )
            }
            Literal(literal) => {
                let block_ref = arena.alloc_block();
                let reg = arena.alloc_reg();
                let instruction = Instruction::new_load_imm(reg, literal.into());
                arena.get_block_mut(block_ref).push(instruction);
                (block_ref.into(), reg)
            }
            Loc(loc) => match *loc {
                Scalar(var) => {
                    let block_ref = arena.alloc_block();
                    let reg = arena.alloc_reg();
                    let instruction = Instruction::new_load(reg, var.into_val());
                    arena.get_block_mut(block_ref).push(instruction);
                    (block_ref.into(), reg)
                }
                Index { arr, size, index } => {
                    let (index_subcfg, index_reg) = index.destruct(arena);
                    let block_ref = arena.alloc_block();
                    let bound_check_instruction = Instruction::new_bound_check(index_reg, size);
                    let reg = arena.alloc_reg();
                    let load_instruction =
                        Instruction::new_load_offset(reg, arr.into_val(), index_reg);
                    arena.get_block_mut(block_ref).push(bound_check_instruction);
                    arena.get_block_mut(block_ref).push(load_instruction);
                    (index_subcfg.concat(block_ref.into(), arena), reg)
                }
            },
            Call(call) => {
                let (cfg, res) = call.destruct(arena);
                (cfg, res.unwrap())
            }
        }
    }
}

impl<'a> HIRStmt<'a> {
    fn destruct(self, arena: &mut Arena, escapes: Escapes) -> SubCfg {
        use HIRStmt::*;
        match self {
            Expr(expr) => {
                let (subcfg, _) = expr.destruct(arena);
                subcfg
            }
            Break => escapes.r#break.unwrap().into(),
            Continue => escapes.r#continue.unwrap().into(),
            Return(expr) => expr.map_or_else(
                || escapes.r#return.into(),
                |expr| {
                    let (subcfg, expr_reg) = expr.destruct(arena);
                    let block_ref = arena.alloc_block();
                    let instruction = Instruction::new_return(expr_reg);
                    arena.get_block_mut(block_ref).push(instruction);
                    subcfg.concat(block_ref.into(), arena)
                },
            ),
            Assign(assign) => assign.destruct(arena),
            If { cond, yes, no } => {
                let (cond_subcfg, cond_reg) = cond.destruct(arena);
                let yes_subcfg = yes.destruct(arena, escapes);
                let no_subcfg = no.destruct(arena, escapes);
                SubCfg::new_if((cond_subcfg, cond_reg), yes_subcfg, no_subcfg, arena)
            }
            While { cond, body } => {
                let (cond_subcfg, cond_reg) = cond.destruct(arena);
                SubCfg::new_loop((cond_subcfg, cond_reg), *body, arena, escapes)
            }
            For {
                init,
                cond,
                update,
                mut body,
            } => {
                let init_subcfg = init.destruct(arena);
                body.stmts.push(update.into());
                let (cond_subcfg, cond_reg) = cond.destruct(arena);
                let r#loop = SubCfg::new_loop((cond_subcfg, cond_reg), *body, arena, escapes);
                init_subcfg.concat(r#loop, arena)
            }
        }
    }
}

impl<'a> HIRBlock<'a> {
    fn destruct(self, arena: &mut Arena, escapes: Escapes) -> SubCfg {
        let decls_block = arena.alloc_block();
        arena
            .get_block_mut(decls_block)
            .extend(self.decls.into_keys().map(Instruction::new_stack_alloc));
        let mut block_subcfg = SubCfg::from(decls_block);
        for stmt in self.stmts {
            let stmt_subcfg = stmt.destruct(arena, escapes);
            block_subcfg = block_subcfg.concat(stmt_subcfg, arena);
        }
        block_subcfg
    }
}

impl<'a> HIRAssign<'a> {
    fn destruct(self, arena: &mut Arena) -> SubCfg {
        use HIRLoc::*;
        let HIRAssign { lhs, rhs } = self;
        let (rhs_subcfg, rhs_reg) = rhs.destruct(arena);
        match lhs {
            Scalar(var) => {
                let block_ref = arena.alloc_block();
                let store_instruction = Instruction::new_store(var.into_val(), rhs_reg);
                arena.get_block_mut(block_ref).push(store_instruction);
                rhs_subcfg.concat(block_ref.into(), arena)
            }
            Index { arr, size, index } => {
                let (index_subcfg, index_reg) = index.destruct(arena);
                let block_ref = arena.alloc_block();
                let bound_check_instruction = Instruction::new_bound_check(index_reg, size);
                let store_instruction =
                    Instruction::new_store_offset(arr.into_val(), index_reg, rhs_reg);
                arena.get_block_mut(block_ref).push(bound_check_instruction);
                arena.get_block_mut(block_ref).push(store_instruction);
                rhs_subcfg
                    .concat(index_subcfg, arena)
                    .concat(block_ref.into(), arena)
            }
        }
    }
}

impl<'a> HIRCall<'a> {
    fn destruct(self, arena: &mut Arena) -> (SubCfg, Option<Reg>) {
        use HIRCall::*;
        match self {
            Extern { .. } => todo!(),
            Decaf { name, ret, args } => {
                let beg = arena.alloc_block();
                let mut args_cfg = SubCfg::from(beg);
                let mut args_regs: Vec<Reg> = vec![];
                for arg in args {
                    let (arg_cfg, arg_reg) = arg.destruct(arena);
                    args_cfg = args_cfg.concat(arg_cfg, arena);
                    args_regs.push(arg_reg);
                }
                if ret.is_some() {
                    let block_ref = arena.alloc_block();
                    let reg = arena.alloc_reg();
                    let instruction = Instruction::new_ret_call(reg, name, args_regs);
                    arena.get_block_mut(block_ref).push(instruction);
                    (args_cfg.concat(block_ref.into(), arena), Some(reg))
                } else {
                    let block_ref = arena.alloc_block();
                    let instruction = Instruction::new_void_call(name, args_regs);
                    arena.get_block_mut(block_ref).push(instruction);
                    (block_ref.into(), None)
                }
            }
        }
    }
}

impl<'a> HIRFunction<'a> {
    pub fn destruct(self) -> Cfg {
        let mut arena = Arena::new();
        if self.ret.is_some() {
            let return_escape = arena.alloc_block();
            let instruction = Instruction::ReturnGuard;
            arena.get_block_mut(return_escape).push(instruction);
            let SubCfg { beg, end } = self.body.destruct(&mut arena, Escapes::new(return_escape));
            Cfg {
                beg,
                end,
                arena,
                name: self.name.to_string(),
            }
        } else {
            let return_escape = arena.alloc_block();
            let SubCfg { beg, end } = self.body.destruct(&mut arena, Escapes::new(return_escape));
            Cfg {
                beg,
                end,
                arena,
                name: self.name.to_string(),
            }
        }
    }
}
