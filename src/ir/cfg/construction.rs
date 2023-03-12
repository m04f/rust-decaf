/// TODO:use registers instead of symbols and hold symbol data somewhere.
use crate::ir::{BBMetaData, BBType::*, BasicBlock, Cfg, Reg, Type};
use std::{marker::PhantomData as Marker, ptr::NonNull};

/// C[onstruction]Cfg a pre-cfg that has a messed up ownership.
struct CCfg<'a> {
    beg: NonNull<BasicBlock>,
    end: Option<NonNull<BasicBlock>>,
    _marker: Marker<&'a BasicBlock>,
}

struct LoopExit<'a> {
    r#continue: CCfg<'a>,
    r#break: CCfg<'a>,
    cond: Reg,
}

pub struct Continue<'a>(NonNull<BasicBlock>, &'a LoopExit<'a>);
pub struct Break<'a>(NonNull<BasicBlock>, &'a LoopExit<'a>);

impl<'a> LoopExit<'a> {
    pub fn r#continue(&'a self) -> Continue<'a> {
        Continue(self.r#continue.beg, self)
    }

    pub fn r#break(&'a self) -> Break<'a> {
        Break(self.r#break.beg, self)
    }

    pub fn new(r#continue: CCfg<'a>, cond: Reg) -> Self {
        Self {
            r#continue,
            r#break: CCfg::new_empty(),
            cond,
        }
    }

    pub fn build(mut self, mut body: CCfg<'a>) -> CCfg<'a> {
        body.try_append_continue(self.r#continue());
        self.r#continue.append_cond(self.cond, body, self.r#break);
        self.r#continue
    }
}

impl<'a> CCfg<'a> {
    /// constructs a new CCfg from beg. panics if `beg` has a terminator.
    pub fn new(beg: Box<BasicBlock>) -> Self {
        let beg = NonNull::new(Box::into_raw(beg)).unwrap();
        Self {
            beg,
            end: Some(beg),
            _marker: Marker,
        }
    }

    pub fn new_empty() -> Self {
        Self::new(Box::new(BBMetaData::new(None).build_block(&[])))
    }

    pub fn can_append(&self) -> bool {
        self.end.is_some()
    }

    pub fn build_graph(self) -> Cfg {
        let mut graph = Cfg::new(self.beg);

        // drop unnecssary nodes
        let mut bfs = graph.bfs_mut();
        while let Some(mut node) = bfs.next() {
            // we do not visit the same node twice so we have to consume as mutch as possible.
            while node.as_mut().merge_tail_if(|tail| tail.incoming_len() == 1) {}
        }

        graph
    }

    pub fn append(&mut self, other: Self) -> &mut Self {
        unsafe {
            if let Some(end) = self.end {
                (*end.as_ptr()).link_unconditional(other.beg);
                self.end = other.end;
                self
            } else {
                unreachable!()
            }
        }
    }

    pub fn append_cond(&mut self, cond: Reg, yes: Self, no: Self) -> &mut Self {
        unsafe {
            if let Some(end) = self.end {
                let join = || {
                    NonNull::new(Box::into_raw(Box::new(
                        BBMetaData::from(CondJoin).build_block(&[]),
                    )))
                    .unwrap()
                };
                let mut yes_start =
                    Self::new(Box::new(BBMetaData::from(IfTrueBranch).build_block(&[])));
                yes_start.append(yes);
                let yes = yes_start;
                let mut no_start =
                    Self::new(Box::new(BBMetaData::from(ElseBranch).build_block(&[])));
                no_start.append(no);
                let no = no_start;

                (*end.as_ptr()).link_conditional(cond, yes.beg, no.beg);

                match (yes.end, no.end) {
                    (None, None) => self.end = None,
                    (Some(end), None) => self.end = Some(end),
                    (None, Some(end)) => self.end = Some(end),
                    (Some(yes_end), Some(no_end)) => {
                        let join = join();
                        (*yes_end.as_ptr()).link_unconditional(join);
                        (*no_end.as_ptr()).link_unconditional(join);
                        self.end = Some(join);
                    }
                }
            } else {
                unreachable!()
            }
            self
        }
    }

    pub fn append_continue(&mut self, r#continue: Continue) {
        unsafe {
            (*self.end.unwrap().as_ptr()).link_unconditional(r#continue.0);
            self.end = None
        }
    }

    pub fn try_append_continue(&mut self, r#continue: Continue) -> bool {
        if let Some(end) = self.end {
            unsafe {
                (*end.as_ptr()).link_unconditional(r#continue.0);
                self.end = None
            }
            true
        } else {
            false
        }
    }

    pub fn append_break(&mut self, r#break: Break) {
        unsafe {
            (*self.end.unwrap().as_ptr()).link_unconditional(r#break.0);
            self.end = None
        }
    }
}

mod destruct {
    use std::collections::HashMap;

    use super::*;
    use crate::hir::*;
    use crate::ir::*;

    pub struct RegSymMap {
        /// maps a symbol with it's id to it's assigned registers.
        assigned_registers: HashMap<(String, usize), Vec<Reg>>,
        /// maps from a register number to the symbol and id.
        registers: Vec<Option<(String, usize)>>,
        id_stack: Vec<usize>,
        next_id: usize,
    }

    impl Extend<((String, usize), Vec<Reg>)> for RegSymMap {
        fn extend<T: IntoIterator<Item = ((String, usize), Vec<Reg>)>>(&mut self, iter: T) {
            for (sym, regs) in iter {
                self.assigned_registers.insert(sym, regs);
            }
        }
    }

    impl RegSymMap {
        fn new() -> Self {
            Self {
                assigned_registers: HashMap::new(),
                registers: Vec::new(),
                id_stack: Vec::new(),
                next_id: 0,
            }
        }

        fn push_id(&mut self) -> usize {
            let id = self.next_id;
            self.id_stack.push(id);
            self.next_id += 1;
            id
        }

        fn pop_id(&mut self) {
            self.next_id = self.id_stack.pop().unwrap();
        }

        /// panics if the symbol is not found.
        fn get(&self, symbol: &String) -> &Vec<Reg> {
            self.assigned_registers
                .get(&(symbol.clone(), *self.id_stack.last().unwrap()))
                .unwrap()
        }

        fn get_mut(&mut self, symbol: &String) -> &mut Vec<Reg> {
            self.id_stack
                .iter()
                .rev()
                .find_map(|id| self.assigned_registers.get_mut(&(symbol.clone(), *id)))
                .unwrap()
        }

        fn unnamed_reg(&mut self) -> Reg {
            self.registers.push(None);
            Reg::Local(self.registers.len() - 1)
        }

        fn nammed_reg(&mut self, symbol: String) -> Reg {
            self.registers
                .push(Some((symbol, *self.id_stack.last().unwrap())));
            Reg::Local(self.registers.len() - 1)
        }
    }

    /// a struct that abstracts away the complexity of `RegSymMap` from the user.
    struct Scope<'a> {
        symbols: &'a mut RegSymMap,
    }

    enum Location {
        Symbol(String),
        Offset(String, Reg),
        Reg(Reg),
    }

    impl From<String> for Location {
        fn from(value: String) -> Self {
            Self::Symbol(value)
        }
    }

    impl From<(String, Reg)> for Location {
        fn from((sym, offset): (String, Reg)) -> Self {
            Self::Offset(sym, offset)
        }
    }

    impl From<Reg> for Location {
        fn from(value: Reg) -> Self {
            Self::Reg(value)
        }
    }

    impl Drop for Scope<'_> {
        fn drop(&mut self) {
            self.symbols.pop_id();
        }
    }

    impl<'a> Scope<'a> {
        fn new<S: ToString>(&mut self, symbols: impl IntoIterator<Item = S>) -> Self {
            let id = self.symbols.push_id();
            self.symbols.extend(
                symbols
                    .into_iter()
                    .map(|sym| ((sym.to_string(), id), vec![])),
            );
            Self {
                symbols: self.symbols,
            }
        }

        fn load_sym(&mut self, symbol: impl ToString) -> (Reg, Instruction) {
            let symbol = symbol.to_string();
            let assigned_registers = self.symbols.get_mut(&symbol);
            if assigned_registers.is_empty() {
                let reg = self.symbols.unnamed_reg();
                let instruction = Instruction::Move {
                    dest: reg.into(),
                    source: Source::UnInit,
                };
                (reg, instruction)
            } else {
                // NOTE: this is terribly inefficent.
                let phi_choices = assigned_registers.iter().map(|&reg| reg.into()).collect();
                let reg = self.symbols.unnamed_reg();
                let instruction = Instruction::Phi {
                    dest: reg.into(),
                    sources: phi_choices,
                };
                (reg, instruction)
            }
        }

        fn load_imm(&mut self, imm: impl Into<Immediate>) -> (Reg, Instruction) {
            let reg = self.symbols.unnamed_reg();
            let instruction = Instruction::Move {
                dest: reg.into(),
                source: imm.into().into(),
            };
            (reg, instruction)
        }

        fn store(&mut self, symbol: impl ToString, reg: Reg) -> Instruction {
            let symbol = symbol.to_string();
            let store_reg = self.symbols.unnamed_reg();
            self.symbols.get_mut(&symbol).push(reg);
            let instruction = Instruction::Move {
                dest: store_reg.into(),
                source: reg.into(),
            };
            instruction
        }

        fn add(&mut self, lhs: Reg, rhs: Reg) -> (Reg, Instruction) {
            let reg = self.symbols.unnamed_reg();
            let instruction = Instruction::Add {
                dest: reg.into(),
                lhs: lhs.into(),
                rhs: rhs.into(),
            };
            (reg, instruction)
        }

        fn sub(&mut self, lhs: Reg, rhs: Reg) -> (Reg, Instruction) {
            let reg = self.symbols.unnamed_reg();
            let instruction = Instruction::Sub {
                dest: reg.into(),
                lhs: lhs.into(),
                rhs: rhs.into(),
            };
            (reg, instruction)
        }
    }

    impl<'a> HIRExpr<'a> {
        fn destruct(&self, reg_allocator: &mut Scope) -> (CCfg, Reg) {
            use HIRExpr::*;
            match self {
                &HIRExpr::Literal(lit) => {
                    let (res, instruction) = reg_allocator.load_imm(lit);
                    let bb = BasicBlock::from_instruction(instruction);
                    (CCfg::new(Box::from(bb)), res)
                }

                &HIRExpr::Len(size) => {
                    let (res, instruction) = reg_allocator.load_imm(size as i64);
                    let bb = BasicBlock::from_instruction(instruction);
                    (CCfg::new(Box::from(bb)), res)
                }

                HIRExpr::Loc(loc) => match loc.as_ref() {
                    HIRLoc::Scalar(sym) => {
                        let (res, instruction) = reg_allocator.load_sym(*sym.val());
                        let bb = BasicBlock::from_instruction(instruction);
                        (CCfg::new(Box::from(bb)), res)
                    }

                    HIRLoc::Index { arr, index, .. } => {
                        let (mut index_ccfg, index) = index.destruct(reg_allocator);
                        let res = reg_allocator.alloc(None);
                        index_ccfg.append(CCfg::new(Box::new(BasicBlock::new(
                            BBMetaData::new(None),
                            &[Instruction::new_load_offset(res, *arr.val(), index)],
                        ))));
                        (index_ccfg, res)
                    }
                },
                HIRExpr::Arith { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator);
                    let res = reg_allocator.alloc(None);
                    let equ = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_arith(res, lhs, *op, rhs)],
                    )));
                    ccfg_lhs.append(ccfg_rhs).append(equ);
                    (ccfg_lhs, res)
                }

                HIRExpr::Rel { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator);
                    let res = reg_allocator.alloc(None);
                    let equ = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_rel(res, lhs, *op, rhs)],
                    )));
                    ccfg_lhs.append(ccfg_rhs).append(equ);
                    (ccfg_lhs, res)
                }
                Eq { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator);
                    let res = reg_allocator.alloc(None);
                    let equ = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_eq(res, lhs, *op, rhs)],
                    )));
                    ccfg_lhs.append(ccfg_rhs).append(equ);
                    (ccfg_lhs, res)
                }
                Neg(e) => {
                    let (mut ccfg, res_neg) = e.destruct(reg_allocator);
                    let res = reg_allocator.alloc(None);
                    let bb = BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_neg(res, res_neg)],
                    );
                    ccfg.append(CCfg::new(Box::new(bb)));
                    (ccfg, res)
                }
                Not(e) => {
                    let (mut ccfg, res_not) = e.destruct(reg_allocator);
                    let res = reg_allocator.alloc(None);
                    let bb = BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_not(res, res_not)],
                    );
                    ccfg.append(CCfg::new(Box::new(bb)));
                    (ccfg, res)
                }
                Ter { cond, yes, no } => {
                    let (mut ccfg_yes, yes) = yes.destruct(reg_allocator);
                    let (ccfg_no, no) = no.destruct(reg_allocator);
                    let (ccfg_cond, cond) = cond.destruct(reg_allocator);
                    let res = reg_allocator.alloc(None);
                    let select = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_select(res, cond, yes, no)],
                    )));
                    ccfg_yes.append(ccfg_no).append(ccfg_cond).append(select);
                    (ccfg_yes, res)
                }
                Call(call) => call.destruct_ret(reg_allocator),
                Cond { op, lhs, rhs } => {
                    let (res, sc) = reg_allocator.alloc_sc();
                    let set_false = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(sc, Immediate::Bool(false))],
                    )));
                    let set_true = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(sc, Immediate::Bool(true))],
                    )));
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator);
                    let (mut ccfg_rhs, rhs) = rhs.destruct(reg_allocator);
                    match op {
                        CondOp::Or => {
                            ccfg_rhs.append_cond(rhs, set_true, set_false);
                            ccfg_lhs.append_cond(lhs, set_true, ccfg_rhs);
                            (ccfg_lhs, res)
                        }
                        CondOp::And => {
                            ccfg_rhs.append_cond(rhs, set_false, set_true);
                            ccfg_lhs.append_cond(lhs, ccfg_rhs, set_false);
                            (ccfg_lhs, res)
                        }
                    }
                }
            }
        }
    }

    impl<'a> HIRCall<'a> {
        fn destruct_ret(&self, reg_allocator: &mut Scope) -> (CCfg, Reg) {
            match self {
                HIRCall::Decaf { name, args, .. } => {
                    let (mut ccfg, args) = args.iter().map(|arg| arg.destruct(reg_allocator)).fold(
                        (CCfg::new_empty(), Vec::new()),
                        |(mut ccfg, mut args), (ccfg_arg, arg)| {
                            ccfg.append(ccfg_arg);
                            args.push(arg);
                            (ccfg, args)
                        },
                    );
                    let res = reg_allocator.alloc(None);
                    let call = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_ret_call(res, *name, args)],
                    )));
                    ccfg.append(call);
                    (ccfg, res)
                }
                HIRCall::Extern { .. } => todo!(),
            }
        }
    }

    impl<'a> HIRStmt<'a> {
        fn destruct(&self, reg_allocator: &mut Scope, loop_exit: Option<&LoopExit>) -> CCfg {
            use HIRStmt::*;
            match self {
                Expr(e) => {
                    let (ccfg, _) = e.destruct(reg_allocator);
                    ccfg
                }
                Break => {
                    let mut ccfg = CCfg::new_empty();
                    ccfg.append_break(loop_exit.unwrap().r#break());
                    ccfg
                }
                Continue => {
                    let mut ccfg = CCfg::new_empty();
                    ccfg.append_continue(loop_exit.unwrap().r#continue());
                    ccfg
                }
                Assign(assign) => assign.destruct(reg_allocator),

                Return(Some(e)) => {
                    let (mut ccfg, res) = e.destruct(reg_allocator);
                    let instruction = Instruction::new_return(res);
                    let bb = BasicBlock::new(BBMetaData::new(None), &[instruction]);
                    ccfg.append(CCfg::new(Box::new(bb)));
                    ccfg
                }
                Return(None) => CCfg::new(Box::new(BasicBlock::new(
                    BBMetaData::new(None),
                    &[Instruction::new_void_ret()],
                ))),
                If { cond, yes, no } => {
                    let (mut ccfg_cond, cond) = cond.destruct(reg_allocator);
                    let ccfg_yes = yes.destruct(reg_allocator, loop_exit);
                    let ccfg_no = no.destruct(reg_allocator, loop_exit);
                    ccfg_cond.append_cond(cond, ccfg_yes, ccfg_no);
                    ccfg_cond
                }
                While { cond, body } => {
                    let (ccfg_cond, cond) = cond.destruct(reg_allocator);
                    let loop_exit = LoopExit::new(ccfg_cond, cond);
                    let body = body.destruct(reg_allocator, Some(&loop_exit));
                    loop_exit.build(body)
                }
                For {
                    init,
                    cond,
                    update,
                    body,
                } => {
                    let mut init = init.destruct(reg_allocator);
                    let (ccfg_cond, cond) = cond.destruct(reg_allocator);
                    let loop_exit = LoopExit::new(ccfg_cond, cond);

                    let update = update.destruct(reg_allocator);
                    let mut body = body.destruct(reg_allocator, Some(&loop_exit));
                    body.append(update);
                    let r#loop = loop_exit.build(body);

                    init.append(r#loop);
                    init
                }
            }
        }
    }

    impl<'a> HIRAssign<'a> {
        fn destruct(&self, reg_allocator: &mut Scope) -> CCfg {
            let (mut ccfg, rhs) = self.rhs.destruct(reg_allocator);
            match &self.lhs {
                HIRLoc::Index { arr, index, .. } => {
                    let (index_ccfg, index) = index.destruct(reg_allocator);
                    let store = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store_offset(*arr.val(), index, rhs)],
                    )));
                    ccfg.append(index_ccfg).append(store);
                    ccfg
                }
                HIRLoc::Scalar(sym) => {
                    let store = Instruction::new_store(*sym.val(), rhs);
                    ccfg.append(CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[store],
                    ))));
                    ccfg
                }
            }
        }
    }

    impl<'a> HIRBlock<'a> {
        fn destruct(&self, reg_allocator: &mut Scope, loop_exit: Option<&LoopExit>) -> CCfg {
            self.stmts
                .iter()
                .map(|stmt| stmt.destruct(reg_allocator, loop_exit))
                .try_fold(CCfg::new_empty(), |mut ccfg, stmt| {
                    ccfg.append(stmt);
                    if ccfg.can_append() {
                        Ok(ccfg)
                    } else {
                        Err(ccfg)
                    }
                })
                .unwrap_or_else(|ccfg| ccfg)
        }
    }

    impl<'a> HIRFunction<'a> {
        pub fn destruct(&self, globals: &[Global]) -> Function {
            let mut graph = self
                .body
                .destruct(&mut Scope::new(globals.len()), None)
                .build_graph();

            // put return guards on the leafs.
            let mut bfs = graph.bfs_mut();
            if self.ret.is_some() {
                while let Some(mut node) = bfs.next() {
                    if node.as_ref().is_leaf() {
                        node.as_mut()
                            .instructions_mut()
                            .push(Instruction::ReturnGuard)
                    }
                }
            }

            Function::new(
                self.name,
                graph,
                self.args_sorted.iter().map(|arg| arg.to_string()).collect(),
            )
        }
    }

    impl HIRRoot<'_> {
        /// converts the HIR tree into an IR tree.
        pub fn destruct(&self) -> Program {
            let globals = self
                .globals
                .values()
                .map(|var| match var {
                    HIRVar::Array { arr, size } => Global::Array {
                        name: arr.val().to_string(),
                        r#type: if arr.is_int() { Type::Int } else { Type::Bool },
                        size: (*size) as usize,
                    },
                    HIRVar::Scalar(scalar) => Global::Scalar {
                        name: scalar.val().to_string(),
                        r#type: if scalar.is_int() {
                            Type::Int
                        } else {
                            Type::Bool
                        },
                    },
                })
                .collect::<Vec<_>>();

            let functions = self
                .functions
                .values()
                .map(|func| func.destruct(&globals))
                .collect();

            Program::new(globals, functions)
        }
    }
}
