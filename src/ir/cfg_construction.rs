use crate::ir::{BBMetaData, BBType::*, BasicBlock, Graph, Reg};
use std::{marker::PhantomData as Marker, ptr::NonNull};

pub struct CCfg<'b> {
    beg: NonNull<BasicBlock<'b>>,
    end: Option<NonNull<BasicBlock<'b>>>,
    _marker: Marker<&'b BasicBlock<'b>>,
}

impl<'a> From<BasicBlock<'a>> for CCfg<'a> {
    fn from(value: BasicBlock<'a>) -> Self {
        CCfg::new(Box::new(value))
    }
}

pub struct LoopExit<'b> {
    r#continue: CCfg<'b>,
    r#break: CCfg<'b>,
    update: Option<CCfg<'b>>,
    cond: Reg,
}

pub struct Continue<'b>(NonNull<BasicBlock<'b>>);
pub struct Break<'b>(NonNull<BasicBlock<'b>>);

impl<'b> LoopExit<'b> {
    pub fn r#continue(&self) -> Continue<'b> {
        self.update
            .as_ref()
            .map_or(Continue(self.r#continue.beg), |update| Continue(update.beg))
    }

    pub fn r#break(&self) -> Break<'b> {
        Break(self.r#break.beg)
    }

    pub fn new(r#continue: CCfg<'b>, cond: Reg, mut update: Option<CCfg<'b>>) -> LoopExit<'b> {
        if let Some(update) = update.as_mut() {
            update.try_append_continue(Continue(r#continue.beg));
        }
        LoopExit {
            r#continue,
            r#break: CCfg::new_empty(),
            update,
            cond,
        }
    }

    pub fn build(mut self, mut body: CCfg<'b>) -> CCfg<'b> {
        body.try_append_continue(self.r#continue());
        self.r#continue.append_cond(self.cond, body, self.r#break);
        self.r#continue
    }
}

impl<'b> CCfg<'b> {
    /// constructs a new CCfg from beg. panics if `beg` has a terminator.
    pub fn new(beg: Box<BasicBlock<'b>>) -> Self {
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

    pub fn build_graph(self) -> Graph<'b> {
        let mut graph = Graph::new(self.beg);
        graph.shrink();
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

    pub fn append_continue(&mut self, r#continue: Continue<'b>) {
        unsafe {
            (*self.end.unwrap().as_ptr()).link_unconditional(r#continue.0);
            self.end = None
        }
    }

    pub fn try_append_continue(&mut self, r#continue: Continue<'b>) -> bool {
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

    pub fn append_break(&mut self, r#break: Break<'b>) {
        unsafe {
            (*self.end.unwrap().as_ptr()).link_unconditional(r#break.0);
            self.end = None
        }
    }
}

mod destruct {
    use super::*;
    use crate::{
        hir::*,
        ir::{
            BBMetaData, BasicBlock, Function, IRExternArg, Immediate, Instruction, Loc, Program,
            Reg, RegAllocator, Symbol,
        },
    };
    use std::collections::HashSet;
    use std::{cell::UnsafeCell, num::NonZeroU16};

    /// A struct that is used to generate unique names for variables. This struct is unsafe +
    /// unsound. although it is wrapped in a safe wrapper for simplicity. so this should never be
    /// exposed to public.
    /// TODO: rename this.
    struct NameMangler<'p, 'b> {
        /// The id used to generate unique names for each variable. This id starts from 1. zero is
        /// reserved for global variables.
        id: u16,
        /// The set of names that live in the scope of the mangler.
        names: HashSet<&'b str>,
        /// A refrence to the parent mangler (scope).
        /// NOTE: this is not a const refrence since we use internal mutability. We actually update
        /// the next field of the parent before we drop the mangler.
        parent: Option<&'p Self>,
        /// A pointer to the next id. This is used to generate unique names for each variable.
        next: UnsafeCell<u16>,
        /// holds the number of fake variables allocated in the stack. The variable starts from id
        /// 1. This does not collide with actual variables since we use `.__st` as the name of the
        /// variable (which is illegal). we need to do this since we do not have an `SSA`
        /// instruction set.
        cooked_vars: UnsafeCell<NonZeroU16>,
    }

    impl<'p, 'b> Drop for NameMangler<'p, 'b> {
        fn drop(&mut self) {
            if let Some(parent) = self.parent.as_mut() {
                unsafe {
                    *parent.next.get() = *self.next.get();
                    *parent.cooked_vars.get() = *self.cooked_vars.get();
                }
            }
        }
    }

    impl<'p, 'b> NameMangler<'p, 'b> {
        /// starts a new nested mangler (scope). This method is unsafe and there can be atmost one
        /// nested mangler at a time.
        fn nest<'s>(&'s self, names: impl IntoIterator<Item = &'b str>) -> NameMangler<'s, 'b>
        where
            's: 'p,
        {
            // NOTE: we do not update the next field in the parent ( we update it in the drop
            // function ).
            unsafe {
                let next = *self.next.get() + 1;
                Self {
                    cooked_vars: UnsafeCell::new(*self.cooked_vars.get()),
                    id: *self.next.get(),
                    names: names.into_iter().collect(),
                    parent: Some(self),
                    next: UnsafeCell::new(next),
                }
            }
        }

        pub fn new(names: impl IntoIterator<Item = &'b str>) -> Self {
            Self {
                cooked_vars: UnsafeCell::new(NonZeroU16::new(1).unwrap()),
                // id 0 is reserved for global variables
                id: 1,
                names: names.into_iter().collect(),
                parent: None,
                next: 2.into(),
            }
        }

        fn mangle(&self, name: &'b str) -> Symbol<'b> {
            if self.names.contains(name) {
                Symbol(name, self.id)
            } else {
                self.parent
                    .as_ref()
                    .map(|par| par.mangle(name))
                    // if it does not exist then it is a global variable
                    .unwrap_or(Symbol(name, 0))
            }
        }

        fn cook_var(&mut self) -> Symbol<'b> {
            unsafe {
                // we use an illegal symbol name to avoid collisions with actual variables.
                let sym = Symbol(".__st", (*self.cooked_vars.get()).get());
                *self.cooked_vars.get() = (*self.cooked_vars.get()).checked_add(1).unwrap();
                sym
            }
        }
    }

    impl<'b> HIRExpr<'b> {
        fn destruct(
            &self,
            reg_allocator: &mut RegAllocator,
            mangler: &mut NameMangler<'_, 'b>,
            func_name: &'b str,
        ) -> (CCfg<'b>, Reg) {
            use HIRExpr::*;
            match self {
                HIRExpr::Literal(lit) => {
                    let res = reg_allocator.alloc_ssa();
                    let bb =
                        BasicBlock::new(BBMetaData::new(None), &[Instruction::new_load(res, *lit)]);
                    (CCfg::new(Box::from(bb)), res)
                }

                HIRExpr::Len(size) => {
                    let res = reg_allocator.alloc_ssa();
                    (
                        BasicBlock::from(&[Instruction::new_load(
                            res,
                            Immediate::Int(size.get() as i64),
                        )])
                        .into(),
                        res,
                    )
                }
                HIRExpr::Loc(loc) => {
                    let (mut ccfg, loc) = loc.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let assign = BasicBlock::from(&[Instruction::new_load(res, loc)]).into();
                    ccfg.append(assign);
                    (ccfg, res)
                }
                HIRExpr::Arith { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let equ =
                        BasicBlock::from(&[Instruction::new_arith(res, lhs, *op, rhs)]).into();
                    ccfg_lhs.append(ccfg_rhs).append(equ);
                    (ccfg_lhs, res)
                }

                HIRExpr::Rel { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let equ = BasicBlock::from(&[Instruction::new_rel(res, lhs, *op, rhs)]);
                    ccfg_lhs.append(ccfg_rhs).append(equ.into());
                    (ccfg_lhs, res)
                }
                Eq { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let equ = BasicBlock::from(&[Instruction::new_eq(res, lhs, *op, rhs)]);
                    ccfg_lhs.append(ccfg_rhs).append(equ.into());
                    (ccfg_lhs, res)
                }
                Neg(e) => {
                    let (mut ccfg, res_neg) = e.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let bb = BasicBlock::from(&[Instruction::new_neg(res, res_neg)]);
                    ccfg.append(bb.into());
                    (ccfg, res)
                }
                Not(e) => {
                    let (mut ccfg, res_not) = e.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let bb = BasicBlock::from(&[Instruction::new_not(res, res_not)]);
                    ccfg.append(bb.into());
                    (ccfg, res)
                }
                Ter { cond, yes, no } => {
                    let res = mangler.cook_var();
                    let (mut ccfg_cond, cond) = cond.destruct(reg_allocator, mangler, func_name);
                    ccfg_cond
                        .append(BasicBlock::from(&[Instruction::AllocScalar { name: res }]).into());
                    let (mut ccfg_yes, yes) = yes.destruct(reg_allocator, mangler, func_name);

                    ccfg_yes.append(BasicBlock::from(&[Instruction::new_store(res, yes)]).into());
                    let (mut ccfg_no, no) = no.destruct(reg_allocator, mangler, func_name);

                    ccfg_no.append(BasicBlock::from(&[Instruction::new_store(res, no)]).into());
                    ccfg_cond.append_cond(cond, ccfg_yes, ccfg_no);

                    let res_reg = reg_allocator.alloc_ssa();
                    let assign = BasicBlock::from(&[Instruction::new_load(res_reg, res)]);
                    ccfg_cond.append(assign.into());
                    (ccfg_cond, res_reg)
                }

                Call(call) => call.destruct_ret(reg_allocator, mangler, func_name),
                Cond {
                    lhs,
                    rhs,
                    op: CondOp::Or,
                } => {
                    let sc = mangler.cook_var();
                    let scbb = BasicBlock::from(&[Instruction::AllocScalar { name: sc }]);
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (mut ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let set_true = BasicBlock::from(&[Instruction::new_store(sc, true)]);
                    let set_false = BasicBlock::from(&[Instruction::new_store(sc, false)]);
                    ccfg_rhs.append_cond(rhs, set_true.into(), set_false.into());
                    let set_true = BasicBlock::from(&[Instruction::new_store(sc, true)]);
                    ccfg_lhs.append_cond(lhs, set_true.into(), ccfg_rhs);
                    let mut beg: CCfg = scbb.into();
                    beg.append(ccfg_lhs);
                    let res = reg_allocator.alloc_ssa();
                    let load = BasicBlock::from(&[Instruction::new_load(res, sc)]);
                    beg.append(load.into());
                    (beg, res)
                }
                Cond {
                    op: CondOp::And,
                    lhs,
                    rhs,
                } => {
                    let sc = mangler.cook_var();
                    let scbb = BasicBlock::from(&[Instruction::AllocScalar { name: sc }]);
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (mut ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let set_false = BasicBlock::from(&[Instruction::new_store(sc, false)]);
                    let set_true = BasicBlock::from(&[Instruction::new_store(sc, true)]);
                    ccfg_rhs.append_cond(rhs, set_true.into(), set_false.into());
                    let set_false = BasicBlock::from(&[Instruction::new_store(sc, false)]);
                    ccfg_lhs.append_cond(lhs, ccfg_rhs, set_false.into());
                    let mut beg: CCfg = scbb.into();
                    beg.append(ccfg_lhs);
                    let res = reg_allocator.alloc_ssa();
                    let load = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_load(res, sc)],
                    )));
                    beg.append(load);
                    (beg, res)
                }
            }
        }
    }

    impl<'b> HIRCall<'b> {
        fn destruct_ret(
            &self,
            reg_allocator: &mut RegAllocator,
            mangler: &mut NameMangler<'_, 'b>,
            func_name: &'b str,
        ) -> (CCfg<'b>, Reg) {
            match self {
                HIRCall::Decaf { name, args, .. } => {
                    let (mut ccfg, args) = args
                        .iter()
                        .map(|arg| arg.destruct(reg_allocator, mangler, func_name))
                        .fold(
                            (CCfg::new_empty(), Vec::new()),
                            |(mut ccfg, mut args), (ccfg_arg, arg)| {
                                ccfg.append(ccfg_arg);
                                args.push(arg);
                                (ccfg, args)
                            },
                        );
                    let res = reg_allocator.alloc_ssa();
                    let call = BasicBlock::from(&[Instruction::new_ret_call(res, *name, args)]);
                    ccfg.append(call.into());
                    (ccfg, res)
                }
                HIRCall::Extern { args, name } => {
                    let mut ccfg = CCfg::new_empty();
                    let args = args
                        .iter()
                        .map(|arg| {
                            match arg {
                                ExternArg::String(sym) => Err(*sym),
                                ExternArg::Array(..) => unimplemented!(),
                                ExternArg::Expr(e) => {
                                    Ok(e.destruct(reg_allocator, mangler, func_name))
                                }
                            }
                            .map(|(ccfg_arg, arg)| {
                                ccfg.append(ccfg_arg);
                                IRExternArg::Source(arg.into())
                            })
                            // trim the two quotes.
                            .map_err(|arg| IRExternArg::String(&arg[1..arg.len() - 1]))
                            .unwrap_or_else(|e| e)
                        })
                        .collect();
                    let res = reg_allocator.alloc_ssa();
                    let call = BasicBlock::from(&[Instruction::new_extern_call(res, *name, args)]);
                    ccfg.append(call.into());
                    (ccfg, res)
                }
            }
        }
    }

    impl<'b> HIRStmt<'b> {
        fn destruct(
            &self,
            reg_allocator: &mut RegAllocator,
            mangler: &mut NameMangler<'_, 'b>,
            loop_exit: Option<&LoopExit<'b>>,
            func_name: &'b str,
        ) -> CCfg<'b> {
            use HIRStmt::*;
            match self {
                Expr(e) => {
                    let (ccfg, _) = e.destruct(reg_allocator, mangler, func_name);
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
                Assign(assign) => assign.destruct(reg_allocator, mangler, func_name),

                Return(Some(e)) => {
                    let (mut ccfg, res) = e.destruct(reg_allocator, mangler, func_name);
                    let bb = BasicBlock::from(&[Instruction::new_return(res)]);
                    ccfg.append(CCfg::new(Box::new(bb)));
                    ccfg
                }
                Return(None) => BasicBlock::from(&[Instruction::new_void_ret()]).into(),
                If { cond, yes, no } => {
                    let (mut ccfg_cond, cond) = cond.destruct(reg_allocator, mangler, func_name);
                    let ccfg_yes = yes.destruct(reg_allocator, mangler, loop_exit, func_name);
                    let ccfg_no = no.destruct(reg_allocator, mangler, loop_exit, func_name);
                    ccfg_cond.append_cond(cond, ccfg_yes, ccfg_no);
                    ccfg_cond
                }
                While { cond, body } => {
                    let (ccfg_cond, cond) = cond.destruct(reg_allocator, mangler, func_name);
                    let loop_exit = LoopExit::new(ccfg_cond, cond, None);
                    let body = body.destruct(reg_allocator, mangler, Some(&loop_exit), func_name);
                    loop_exit.build(body)
                }
                For {
                    init,
                    cond,
                    update,
                    body,
                } => {
                    let mut init = init.destruct(reg_allocator, mangler, func_name);
                    let (ccfg_cond, cond) = cond.destruct(reg_allocator, mangler, func_name);
                    let update = update.destruct(reg_allocator, mangler, func_name);
                    let loop_exit = LoopExit::new(ccfg_cond, cond, Some(update));

                    let body = body.destruct(reg_allocator, mangler, Some(&loop_exit), func_name);
                    let r#loop = loop_exit.build(body);

                    init.append(r#loop);
                    init
                }
            }
        }
    }

    impl<'b> HIRLoc<'b> {
        fn destruct(
            &self,
            reg_allocator: &mut RegAllocator,
            mangler: &mut NameMangler<'_, 'b>,
            func_name: &'b str,
        ) -> (CCfg<'b>, Loc<'b>) {
            match &self {
                HIRLoc::Index { arr, index, size } => {
                    let (mut index_ccfg, index) = index.destruct(reg_allocator, mangler, func_name);
                    let less_than_zero = reg_allocator.alloc_ssa();
                    let larger_than_bound = reg_allocator.alloc_ssa();
                    let bound_cond = reg_allocator.alloc_ssa();
                    let mut bound_check: CCfg = BasicBlock::from(&[
                        Instruction::new_rel(less_than_zero, index, RelOp::Less, Immediate::Int(0)),
                        Instruction::new_rel(
                            larger_than_bound,
                            index,
                            RelOp::GreaterEqual,
                            Immediate::Int(size.get() as i64),
                        ),
                        Instruction::new_cond(
                            bound_cond,
                            less_than_zero,
                            CondOp::Or,
                            larger_than_bound,
                        ),
                    ])
                    .into();
                    let on_bound_fail = BasicBlock::from(&[
                        Instruction::new_extern_call(
                            reg_allocator.alloc_ssa(),
                            "printf",
                            vec![
                                IRExternArg::String(
                                    r#"*** RUNTIME ERROR ***: Array out of Bounds access in method \"%s\"\n"#,
                                ),
                                IRExternArg::String(func_name),
                            ],
                        ),
                        Instruction::Exit(-1),
                    ]);
                    bound_check.append_cond(bound_cond, on_bound_fail.into(), CCfg::new_empty());
                    index_ccfg.append(bound_check);
                    (
                        index_ccfg,
                        Loc::Offset(mangler.mangle(arr.into_val().as_str()), index),
                    )
                }
                HIRLoc::Scalar(sym) => (
                    CCfg::new_empty(),
                    mangler.mangle(sym.into_val().as_str()).into(),
                ),
            }
        }
    }

    impl<'b> AssignOp<'b> {
        fn destruct(
            &self,
            reg_allocator: &mut RegAllocator,
            mangler: &mut NameMangler<'_, 'b>,
            func_name: &'b str,
        ) -> (CCfg<'b>, Reg) {
            match self {
                AssignOp::AddAssign(e) | AssignOp::SubAssign(e) | AssignOp::Assign(e) => {
                    e.destruct(reg_allocator, mangler, func_name)
                }
            }
        }
    }

    impl<'b> HIRAssign<'b> {
        fn destruct(
            &self,
            reg_allocator: &mut RegAllocator,
            mangler: &mut NameMangler<'_, 'b>,
            func_name: &'b str,
        ) -> CCfg<'b> {
            let (mut ccfg_loc, lhs) = self.lhs.destruct(reg_allocator, mangler, func_name);
            let (ccfg, rhs) = self.rhs.destruct(reg_allocator, mangler, func_name);
            let assign = BasicBlock::from(&match self.rhs {
                AssignOp::AddAssign(_) => [Instruction::new_arith(lhs, lhs, ArithOp::Add, rhs)],
                AssignOp::SubAssign(_) => [Instruction::new_arith(lhs, lhs, ArithOp::Sub, rhs)],
                AssignOp::Assign(_) => [Instruction::new_store(lhs, rhs)],
            });
            ccfg_loc.append(ccfg).append(assign.into());
            ccfg_loc
        }
    }

    impl<'b> HIRBlock<'b> {
        fn destruct(
            &self,
            reg_allocator: &mut RegAllocator,
            mangler: &mut NameMangler<'_, 'b>,
            loop_exit: Option<&LoopExit<'b>>,
            func_name: &'b str,
        ) -> CCfg<'b> {
            let mut mangler = mangler.nest(self.decls().keys().map(|span| span.as_str()));
            let stack_allocs = self
                .decls()
                .values()
                .flat_map(|decl| {
                    if decl.is_scalar() {
                        let scalar = mangler.mangle(decl.name().as_str());
                        [
                            Instruction::AllocScalar { name: scalar },
                            Instruction::InitSymbol { name: scalar },
                        ]
                    } else {
                        let name = mangler.mangle(decl.name().as_str());
                        [
                            Instruction::AllocArray {
                                name,
                                size: decl.array_len().unwrap(),
                            },
                            Instruction::InitArray {
                                name,
                                size: decl.array_len().unwrap(),
                            },
                        ]
                    }
                })
                .collect::<Vec<_>>();
            let mut ccfg: CCfg = BasicBlock::from(stack_allocs.as_slice()).into();
            for stmt in self.stmts.iter() {
                let stmt = stmt.destruct(reg_allocator, &mut mangler, loop_exit, func_name);
                ccfg.append(stmt);
                if !ccfg.can_append() {
                    break;
                }
            }
            ccfg
        }
    }

    impl<'a> HIRFunction<'a> {
        pub fn destruct(&self) -> Function {
            let mut mangler = NameMangler::new(self.args_sorted.iter().map(|name| name.as_str()));
            let mut ccfg: CCfg = BasicBlock::from(&[]).into();
            let mut reg_allocator = RegAllocator::new();
            let body =
                self.body
                    .destruct(&mut reg_allocator, &mut mangler, None, self.name.as_str());
            ccfg.append(body);
            let graph = ccfg.build_graph();

            let mut func = Function::new(
                self.name.as_str(),
                graph,
                reg_allocator.allocated(),
                self.args_sorted
                    .iter()
                    .map(|arg| mangler.mangle(arg.as_str())),
            );

            // put return guards on the leafs.
            if self.ret.is_some() {
                let end_of_control_rt_error = [
                    Instruction::ExternCall {
                        dest: func.allocate_reg().into(),
                        symbol: "printf",
                        args: vec![
                            IRExternArg::String(
                                r#"*** RUNTIME ERROR ***: No return value from non-void method \"%s\"\n"#,
                            ),
                            IRExternArg::String(func.name()),
                        ],
                    },
                    Instruction::Exit(-2),
                ];
                let mut bfs = func.graph.bfs_mut();
                while let Some(mut node) = bfs.next() {
                    if node.as_ref().is_leaf() {
                        node.as_mut()
                            .instructions_mut()
                            .extend(end_of_control_rt_error.clone())
                    }
                }
            } else {
                let mut bfs = func.graph_mut().bfs_mut();
                while let Some(mut node) = bfs.next() {
                    if node.as_ref().is_leaf() {
                        node.as_mut()
                            .instructions_mut()
                            .push(Instruction::VoidReturn)
                    }
                }
            }

            func
        }
    }

    impl<'a> HIRRoot<'a> {
        pub fn destruct(&self) -> Program {
            Program::new(
                self.globals.values().map(|var| match var {
                    HIRVar::Scalar(name) => (name.into_val().as_str(), 8),
                    HIRVar::Array { arr, size, .. } => {
                        (arr.into_val().as_str(), 8 * size.get() as usize)
                    }
                }),
                self.functions.values().map(|func| func.destruct()),
            )
        }
    }
}
