//! TODO: this module requries alot of refactoring.

mod instruction;
pub use basicblock::*;
pub use dot::*;
pub use graph::*;
pub use instruction::*;

mod basicblock {
    pub use crate::ir::instruction::*;
    use std::{collections::HashSet, ptr::NonNull};
    pub struct RegAllocator(usize);

    impl RegAllocator {
        pub fn new() -> Self {
            Self(0)
        }
        pub fn alloc(&mut self) -> Reg {
            let reg = Reg::new(self.0);
            self.0 += 1;
            reg
        }
    }

    impl Default for RegAllocator {
        fn default() -> Self {
            Self::new()
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) enum Terminator {
        Fork {
            cond: Reg,
            yes: NonNull<BasicBlock>,
            no: NonNull<BasicBlock>,
        },
        Tail {
            block: NonNull<BasicBlock>,
        },
    }

    #[derive(Debug, Clone)]
    pub enum BBType {
        ShortCircuiting,
        IfTrueBranch,
        ElseBranch,
        CondJoin,
        LoopCondition,
        LoopBody,
        LoopExit,
    }

    impl From<BBType> for BBMetaData {
        fn from(value: BBType) -> Self {
            BBMetaData {
                ty: Some(value),
                is_root: false,
            }
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct BBMetaData {
        ty: Option<BBType>,
        is_root: bool,
    }

    impl BBMetaData {
        pub fn new(ty: Option<BBType>) -> Self {
            Self { ty, is_root: false }
        }

        pub fn build_block(self, insrtctions: &[Instruction]) -> BasicBlock {
            BasicBlock::new(self, insrtctions)
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct BasicBlock {
        incoming: HashSet<NonNull<BasicBlock>>,
        instructions: Vec<Instruction>,
        terminator: Option<Terminator>,
        metadata: BBMetaData,
    }

    impl BasicBlock {
        pub(super) fn terminator(&self) -> Option<&Terminator> {
            self.terminator.as_ref()
        }

        pub fn insrtuctions(&self) -> &[Instruction] {
            &self.instructions
        }

        pub fn is_root(&self) -> bool {
            self.metadata.is_root
        }

        /// makes the block a root block. this is unsafe since making a non-root block a root can
        /// cause memory leakes.
        pub(super) unsafe fn make_root(&mut self) {
            self.metadata.is_root = true;
        }

        pub fn instructions_mut(&mut self) -> &mut Vec<Instruction> {
            &mut self.instructions
        }

        pub fn incoming_len(&self) -> usize {
            self.incoming.len()
        }

        pub fn extend(&mut self, other: Self) {
            assert!(self.terminator.is_none());
            self.instructions.extend(other.instructions);
            self.terminator = other.terminator;
        }

        pub fn push(&mut self, instruction: Instruction) {
            self.instructions.push(instruction);
        }

        pub fn new(metadata: BBMetaData, instruction: &[Instruction]) -> Self {
            Self {
                incoming: HashSet::new(),
                instructions: instruction.to_vec(),
                terminator: None,
                metadata,
            }
        }

        pub(super) fn link_unconditional(&mut self, mut next: NonNull<Self>) {
            assert!(self.terminator.is_none());
            unsafe {
                self.terminator = Some(Terminator::Tail { block: next });
                assert!(next
                    .as_mut()
                    .incoming
                    .insert(NonNull::new(self as *mut Self).unwrap()));
            }
        }

        pub(super) fn link_conditional(
            &mut self,
            cond: Reg,
            mut yes: NonNull<Self>,
            mut no: NonNull<Self>,
        ) {
            assert!(self.terminator.is_none());
            unsafe {
                self.terminator = Some(Terminator::Fork { cond, yes, no });
                assert!(yes
                    .as_mut()
                    .incoming
                    .insert(NonNull::new(self as *mut Self).unwrap()));
                assert!(no
                    .as_mut()
                    .incoming
                    .insert(NonNull::new(self as *mut Self).unwrap()));
            }
        }

        /// drops the node if there is no incoming edges.
        ///
        /// NOTE: this does not look for cycles so if there can be cycles in the grpah they should
        /// be handled somewhere else.
        unsafe fn drop_if_needed(mut bb: NonNull<BasicBlock>) {
            unsafe {
                if (*bb.as_ptr()).incoming.is_empty() {
                    bb.as_mut().unlink();
                    // we can not drop root blocks.
                    if !bb.as_ref().is_root() {
                        drop(Box::from_raw(bb.as_ptr()));
                    }
                } else {
                }
            }
        }

        pub fn unlink_tail(&mut self) {
            assert!(self.tail().is_some());
            self.unlink();
        }

        /// drops any outgoing edges
        pub fn unlink(&mut self) {
            match self.terminator() {
                Some(&Terminator::Fork {
                    mut yes, mut no, ..
                }) => unsafe {
                    yes.as_mut()
                        .incoming
                        .remove(&NonNull::new(self as *const Self as *mut Self).unwrap());
                    no.as_mut()
                        .incoming
                        .remove(&NonNull::new(self as *const Self as *mut Self).unwrap());
                    Self::drop_if_needed(yes);
                    Self::drop_if_needed(no);
                },
                Some(&Terminator::Tail { mut block }) => unsafe {
                    block
                        .as_mut()
                        .incoming
                        .remove(&NonNull::new(self as *const _ as *mut _).unwrap());
                    Self::drop_if_needed(block);
                },
                None => {}
            }
            self.terminator = None;
        }

        /// returns the tail of the block
        pub fn tail(&self) -> Option<&BasicBlock> {
            match self.terminator {
                Some(Terminator::Tail { block, .. }) => Some(unsafe { block.as_ref() }),
                _ => None,
            }
        }

        /// merges the tail block if it is a tail block. and `f(tail)` returns true.
        ///
        /// returns `true` if the tail was mereged
        pub fn merge_tail_if(&mut self, p: impl FnOnce(&BasicBlock) -> bool) -> bool {
            if let Some(tail) = self.tail() {
                if p(tail) {
                    self.instructions.extend(tail.instructions.clone());
                    debug_assert!(self.bypass_tail());
                    true
                } else {
                    false
                }
            } else {
                false
            }
        }

        /// bypasses the tail of the block if it is a tail block.
        /// NOTE: This does not mutate the instructions.
        pub fn bypass_tail(&mut self) -> bool {
            if let Some(&Terminator::Tail { block }) = self.terminator() {
                unsafe {
                    // bind the tail node so that it does not drop the subgraph
                    let mut bind = BasicBlock::default();
                    bind.link_unconditional(block);
                    self.unlink_tail();
                    match block.as_ref().terminator() {
                        Some(&Terminator::Tail { block: tail }) => {
                            self.link_unconditional(tail);
                        }
                        Some(&Terminator::Fork { cond, yes, no }) => {
                            self.link_conditional(cond, yes, no);
                        }
                        None => {}
                    }

                    // release the binding
                    bind.unlink_tail();
                    true
                }
            } else {
                false
            }
        }
    }
}

mod graph {
    use crate::ir::*;
    use std::{
        collections::{HashSet, VecDeque},
        fmt::Display,
        marker::PhantomData as Marker,
        ptr::NonNull,
    };

    pub struct Graph {
        root: NonNull<BasicBlock>,
        _marker: Marker<BasicBlock>,
    }

    impl Drop for Graph {
        fn drop(&mut self) {
            let mut dfs = self.dfs_mut();
            while let Some(mut block) = dfs.next() {
                block.as_mut().unlink();
            }

            // since the node is assumed to be visited at the beginning it is never dropped. note
            // that this is a desirable behaviour because if someone destroys the whole graph in a
            // dfs visit the graph should have at least an empty root.
            unsafe {
                drop(Box::from_raw(self.root.as_ptr()));
            }
        }
    }

    pub struct Node<'a> {
        block: NonNull<BasicBlock>,
        _marker: Marker<&'a BasicBlock>,
    }

    pub struct NodeMut<'a> {
        block: NonNull<BasicBlock>,
        _marker: Marker<&'a BasicBlock>,
    }

    impl AsRef<BasicBlock> for Node<'_> {
        fn as_ref(&self) -> &BasicBlock {
            unsafe { self.block.as_ref() }
        }
    }

    impl AsRef<BasicBlock> for NodeMut<'_> {
        fn as_ref(&self) -> &BasicBlock {
            unsafe { self.block.as_ref() }
        }
    }

    impl AsMut<BasicBlock> for NodeMut<'_> {
        fn as_mut(&mut self) -> &mut BasicBlock {
            unsafe { self.block.as_mut() }
        }
    }

    impl NodeMut<'_> {
        pub fn instructions_mut(&mut self) -> &mut Vec<Instruction> {
            self.as_mut().instructions_mut()
        }

        fn terminator(&self) -> Option<&Terminator> {
            self.as_ref().terminator()
        }

        pub fn tail(&self) -> Option<Node> {
            self.as_ref().terminator().and_then(|t| match t {
                &Terminator::Tail { block, .. } => Some(Node {
                    block,
                    _marker: Marker,
                }),
                _ => None,
            })
        }

        /// drops any outgoing edges
        pub fn unlink(&mut self) {
            self.as_mut().unlink()
        }

        pub fn bypass_tail(&mut self) -> bool {
            self.as_mut().bypass_tail()
        }

        pub fn as_node(&self) -> Node {
            Node {
                block: self.block,
                _marker: Marker,
            }
        }

        pub fn is_leaf(&self) -> bool {
            self.as_ref().terminator().is_none()
        }

        pub fn merge_tail_if(&mut self, p: impl FnOnce(&BasicBlock) -> bool) -> bool {
            self.as_mut().merge_tail_if(p)
        }
    }

    pub struct NodeId(usize);

    impl Display for NodeId {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "N{}", self.0)
        }
    }

    /// enum that represents the **out goining** edges from the node.
    pub enum NeighbouringNodes<'a> {
        Unconditional {
            next: Node<'a>,
        },
        Conditional {
            cond: Reg,
            yes: Node<'a>,
            no: Node<'a>,
        },
    }

    impl<'a> Node<'a> {
        fn terminator(&self) -> Option<&Terminator> {
            self.as_ref().terminator()
        }

        pub fn is_leaf(&self) -> bool {
            self.terminator().is_none()
        }

        pub fn neighbours(&self) -> Option<NeighbouringNodes<'a>> {
            match self.as_ref().terminator() {
                None => None,
                Some(&Terminator::Tail { block, .. }) => Some(NeighbouringNodes::Unconditional {
                    next: Node {
                        block,
                        _marker: Marker,
                    },
                }),
                Some(&Terminator::Fork { yes, no, cond, .. }) => {
                    Some(NeighbouringNodes::Conditional {
                        cond,
                        yes: Node {
                            block: yes,
                            _marker: Marker,
                        },
                        no: Node {
                            block: no,
                            _marker: Marker,
                        },
                    })
                }
            }
        }

        /// returns a unique identifier for the basicblock
        pub fn id(&self) -> NodeId {
            NodeId(self.as_ref() as *const _ as usize)
        }
    }

    enum DfsNode {
        Recurse(NonNull<BasicBlock>),
        Shallow(NonNull<BasicBlock>),
    }

    pub struct Dfs<'a> {
        visited: HashSet<NonNull<BasicBlock>>,
        stack: Vec<DfsNode>,
        _marker: Marker<&'a BasicBlock>,
    }

    pub struct DfsMut<'a> {
        visited: HashSet<NonNull<BasicBlock>>,
        stack: Vec<DfsNode>,
        _marker: Marker<&'a mut BasicBlock>,
    }

    #[derive(Debug)]
    struct BfsBookKeeper {
        visited: HashSet<NonNull<BasicBlock>>,
        queue: VecDeque<NonNull<BasicBlock>>,
    }

    /// a bfs iterator that allows mutating nodes. the nodes can mutate two things the instructions
    /// list of a block and the outgoing edges in the current `BfsMutNode`.
    pub struct BfsMut<'a> {
        book_keeper: BfsBookKeeper,
        _marker: Marker<&'a mut BasicBlock>,
    }

    impl BfsBookKeeper {
        /// conditionally adds a node to the queue
        fn enqueue(&mut self, node: NonNull<BasicBlock>) {
            if self.visited.insert(node) {
                self.queue.push_back(node);
            }
        }

        /// returns the next node to be visited (if it exists).
        fn dequeue(&mut self) -> Option<NonNull<BasicBlock>> {
            self.queue.pop_front()
        }

        fn clear(&mut self) {
            assert!(self.queue.is_empty());
            self.visited.clear();
        }
    }

    impl BfsMut<'_> {
        #[allow(clippy::should_implement_trait)]
        pub fn next(&mut self) -> Option<BfsMutNode<'_>> {
            if let Some(top) = self.book_keeper.dequeue() {
                Some(BfsMutNode {
                    node: NodeMut {
                        block: top,
                        _marker: Marker,
                    },
                    book_keeper: &mut self.book_keeper,
                })
            } else {
                // the bfs is empty we clean any allocated memroy.
                self.book_keeper.clear();
                None
            }
        }
    }

    impl DfsMut<'_> {
        #[allow(clippy::should_implement_trait)]
        pub fn next(&mut self) -> Option<NodeMut> {
            if let Some(top) = self.stack.pop() {
                match top {
                    DfsNode::Shallow(node) => {
                        self.visited.insert(node);
                        Some(NodeMut {
                            block: node,
                            _marker: Marker,
                        })
                    }
                    DfsNode::Recurse(node) => match unsafe { node.as_ref() }.terminator() {
                        None => {
                            self.visited.insert(node);
                            Some(NodeMut {
                                block: node,
                                _marker: Marker,
                            })
                        }
                        Some(Terminator::Tail { block, .. }) => {
                            self.stack.push(DfsNode::Shallow(node));
                            // PERF: it is quite hard for the compiler to optimize this
                            self.stack.push(DfsNode::Recurse(*block));
                            self.next()
                        }
                        Some(Terminator::Fork { yes, no, .. }) => {
                            self.stack.push(DfsNode::Shallow(node));
                            self.stack.push(DfsNode::Recurse(*yes));
                            // PERF: it is quite hard for the compiler to optimize this
                            self.stack.push(DfsNode::Recurse(*no));
                            self.next()
                        }
                    },
                }
            } else {
                None
            }
        }
    }

    pub struct BfsMutNode<'a> {
        /// a refrence to the bfs iterator so that we can update the queue when the node is getting
        /// destructed.
        book_keeper: &'a mut BfsBookKeeper,
        node: NodeMut<'a>,
    }

    impl<'a> AsRef<NodeMut<'a>> for BfsMutNode<'a> {
        fn as_ref(&self) -> &NodeMut<'a> {
            &self.node
        }
    }

    impl<'a> AsMut<NodeMut<'a>> for BfsMutNode<'a> {
        fn as_mut(&mut self) -> &mut NodeMut<'a> {
            &mut self.node
        }
    }

    impl<'a> Drop for BfsMutNode<'a> {
        fn drop(&mut self) {
            match self.as_ref().terminator() {
                None => {}
                Some(&Terminator::Tail { block, .. }) => {
                    self.book_keeper.enqueue(block);
                }
                Some(&Terminator::Fork { yes, no, .. }) => {
                    self.book_keeper.enqueue(yes);
                    self.book_keeper.enqueue(no);
                }
            };
        }
    }

    impl<'a> Dfs<'a> {
        fn new(root: &'a BasicBlock) -> Self {
            Self {
                visited: HashSet::default(),
                stack: vec![DfsNode::Recurse(
                    NonNull::new(root as *const _ as *mut _).unwrap(),
                )],
                _marker: Marker,
            }
        }
    }

    impl<'a> Iterator for Dfs<'a> {
        type Item = Node<'a>;
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(top) = self.stack.pop() {
                match top {
                    DfsNode::Shallow(node) => {
                        self.visited.insert(node);
                        Some(Node {
                            block: node,
                            _marker: Marker,
                        })
                    }
                    DfsNode::Recurse(node) => match unsafe { node.as_ref() }.terminator() {
                        None => {
                            self.visited.insert(node);
                            Some(Node {
                                block: node,
                                _marker: Marker,
                            })
                        }
                        Some(Terminator::Tail { block, .. }) => {
                            self.stack.push(DfsNode::Shallow(node));
                            // PERF: it is quite hard for the compiler to optimize this
                            self.stack.push(DfsNode::Recurse(*block));
                            self.next()
                        }
                        Some(Terminator::Fork { yes, no, .. }) => {
                            self.stack.push(DfsNode::Shallow(node));
                            self.stack.push(DfsNode::Recurse(*yes));
                            // PERF: it is quite hard for the compiler to optimize this
                            self.stack.push(DfsNode::Recurse(*no));
                            self.next()
                        }
                    },
                }
            } else {
                None
            }
        }
    }

    impl Graph {
        pub fn new(root: NonNull<BasicBlock>) -> Self {
            let mut graph = Self {
                root,
                _marker: Marker,
            };

            // mark the root so that it does not get dropped.
            unsafe {
                graph.root_mut().make_root();
            }

            graph
        }

        pub fn root(&self) -> &BasicBlock {
            unsafe { self.root.as_ref() }
        }

        pub fn root_mut(&mut self) -> &mut BasicBlock {
            unsafe { self.root.as_mut() }
        }

        pub fn dfs(&self) -> Dfs<'_> {
            Dfs::new(self.root())
        }

        pub fn dfs_mut(&mut self) -> DfsMut<'_> {
            DfsMut {
                visited: HashSet::default(),
                stack: vec![DfsNode::Recurse(self.root)],
                _marker: Marker,
            }
        }

        pub fn bfs_mut(&mut self) -> BfsMut {
            let book_keeper = BfsBookKeeper {
                visited: HashSet::from([self.root]),
                queue: VecDeque::from([self.root]),
            };
            BfsMut {
                book_keeper,
                _marker: Marker,
            }
        }
    }

    pub struct Function {
        graph: Graph,
        name: String,
        args: Vec<String>,
    }

    impl AsRef<Graph> for Function {
        fn as_ref(&self) -> &Graph {
            self.graph()
        }
    }

    impl AsMut<Graph> for Function {
        fn as_mut(&mut self) -> &mut Graph {
            self.graph_mut()
        }
    }

    impl Function {
        pub fn new(name: impl ToString, graph: Graph, args: Vec<String>) -> Self {
            Self {
                name: name.to_string(),
                args,
                graph,
            }
        }

        pub fn args(&self) -> &[String] {
            &self.args
        }

        pub fn to_dot(&self) -> Dot {
            self.as_ref().to_dot(Some(self.name()))
        }

        pub fn name(&self) -> &str {
            self.name.as_str()
        }

        pub fn graph(&self) -> &Graph {
            &self.graph
        }

        pub fn graph_mut(&mut self) -> &mut Graph {
            &mut self.graph
        }
    }
}

mod ccfg {
    use crate::ir::{BBMetaData, BBType::*, BasicBlock, Graph, Reg};
    use std::{marker::PhantomData as Marker, ptr::NonNull};

    pub struct CCfg<'a> {
        beg: NonNull<BasicBlock>,
        end: Option<NonNull<BasicBlock>>,
        _marker: Marker<&'a BasicBlock>,
    }

    pub struct LoopExit<'a> {
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

        pub fn build_graph(self) -> Graph {
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
}

mod destruct {
    use super::ccfg::*;
    use crate::{
        hir::*,
        ir::{BBMetaData, BasicBlock, Function, Immediate, Instruction, Reg, RegAllocator},
    };

    impl<'a> HIRExpr<'a> {
        fn destruct(&self, reg_allocator: &mut RegAllocator) -> (CCfg, Reg) {
            use HIRExpr::*;
            match self {
                HIRExpr::Literal(lit) => {
                    let res = reg_allocator.alloc();
                    let bb = BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_load_imm(res, *lit)],
                    );
                    (CCfg::new(Box::from(bb)), res)
                }

                HIRExpr::Len(size) => {
                    let res = reg_allocator.alloc();
                    (
                        CCfg::new(Box::new(BasicBlock::new(
                            BBMetaData::new(None),
                            &[Instruction::new_load_imm(res, Immediate::Int(*size as i64))],
                        ))),
                        res,
                    )
                }
                HIRExpr::Loc(loc) => match loc.as_ref() {
                    HIRLoc::Scalar(sym) => {
                        let res = reg_allocator.alloc();
                        let bb = BasicBlock::new(
                            BBMetaData::new(None),
                            &[Instruction::new_load(res, *sym.val())],
                        );
                        (CCfg::new(Box::new(bb)), res)
                    }
                    HIRLoc::Index { arr, size, index } => {
                        let (mut index_ccfg, index) = index.destruct(reg_allocator);
                        let res = reg_allocator.alloc();
                        index_ccfg.append(CCfg::new(Box::new(BasicBlock::new(
                            BBMetaData::new(None),
                            &[Instruction::new_load_offset(res, *arr.val(), index, *size)],
                        ))));
                        (index_ccfg, res)
                    }
                },
                HIRExpr::Arith { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator);
                    let res = reg_allocator.alloc();
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
                    let res = reg_allocator.alloc();
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
                    let res = reg_allocator.alloc();
                    let equ = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_eq(res, lhs, *op, rhs)],
                    )));
                    ccfg_lhs.append(ccfg_rhs).append(equ);
                    (ccfg_lhs, res)
                }
                Neg(e) => {
                    let (mut ccfg, res_neg) = e.destruct(reg_allocator);
                    let res = reg_allocator.alloc();
                    let bb = BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_neg(res, res_neg)],
                    );
                    ccfg.append(CCfg::new(Box::new(bb)));
                    (ccfg, res)
                }
                Not(e) => {
                    let (mut ccfg, res_not) = e.destruct(reg_allocator);
                    let res = reg_allocator.alloc();
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
                    let res = reg_allocator.alloc();
                    let select = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_select(res, cond, yes, no)],
                    )));
                    ccfg_yes.append(ccfg_no).append(ccfg_cond).append(select);
                    (ccfg_yes, res)
                }
                Call(call) => call.destruct_ret(reg_allocator),
                Cond { .. } => todo!(),
            }
        }
    }

    impl<'a> HIRCall<'a> {
        fn destruct_ret(&self, reg_allocator: &mut RegAllocator) -> (CCfg, Reg) {
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
                    let res = reg_allocator.alloc();
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
        fn destruct(&self, reg_allocator: &mut RegAllocator, loop_exit: Option<&LoopExit>) -> CCfg {
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
        fn destruct(&self, reg_allocator: &mut RegAllocator) -> CCfg {
            let (mut ccfg, rhs) = self.rhs.destruct(reg_allocator);
            match &self.lhs {
                HIRLoc::Index { arr, size, index } => {
                    let (index_ccfg, index) = index.destruct(reg_allocator);
                    let store = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store_offset(*arr.val(), index, rhs, *size)],
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
        fn destruct(&self, reg_allocator: &mut RegAllocator, loop_exit: Option<&LoopExit>) -> CCfg {
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
        pub fn destruct(&self) -> Function {
            let mut graph = self
                .body
                .destruct(&mut RegAllocator::new(), None)
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

            Function::new(self.name, graph, self.args_sorted.iter().map(|arg| arg.to_string()).collect())
        }
    }
}

mod dot {
    use std::{
        fmt::Display,
        io::{self, Write},
        path::Path,
        process::{Command, ExitStatus, Stdio},
    };

    use crate::ir::{Graph, NeighbouringNodes, Node};

    pub struct Dot(String);

    impl Dot {
        /// compiles the given graph. (it currently compiles to svg image no matter what).
        ///
        /// returns the exit status of the `dot` command.
        pub fn compile(&self, out_file: impl AsRef<Path>) -> io::Result<ExitStatus> {
            let mut dot_proc = Command::new("dot")
                .arg("-Tsvg")
                .arg("-o")
                .arg(out_file.as_ref())
                .stdin(Stdio::piped())
                .spawn()?;
            dot_proc
                .stdin
                .as_ref()
                .unwrap()
                .write_all(self.0.as_bytes())?;
            dot_proc.wait()
        }

        /// displays the graphs in the systems image viewer (uses xdg-open).
        ///
        /// `graphs` is a list of compiled graphs to be displayed.
        ///
        /// returns the exit status of the `xdg-open` command.
        pub fn display<P: AsRef<Path>>(
            graphs: impl IntoIterator<Item = P>,
        ) -> io::Result<ExitStatus> {
            let mut proc = Command::new("xdg-open");
            for graph in graphs {
                proc.arg(graph.as_ref());
            }
            proc.spawn()?.wait()
        }
    }

    impl Node<'_> {
        fn dot_label(&self) -> String {
            self.as_ref()
                .insrtuctions()
                .iter()
                .fold(String::new(), |mut acc, instruction| {
                    acc.push_str(&format!("{instruction:?}\\n"));
                    acc
                })
        }

        /// returns the **out goining** edges
        fn dot_links(&self) -> String {
            match self.neighbours() {
                None => String::default(),
                Some(NeighbouringNodes::Conditional { yes, no, .. }) => {
                    // TODO: print the condition register
                    format!("{} -> {{{} {}}}\n", self.id(), yes.id(), no.id())
                }
                Some(NeighbouringNodes::Unconditional { next }) => {
                    format!("{} -> {}\n", self.id(), next.id())
                }
            }
        }
    }

    impl Graph {
        /// converts the graph to a dot graph.
        pub fn to_dot<T: Display>(&self, name: Option<T>) -> Dot {
            let graph_body = self
                .dfs()
                .map(|node| {
                    format!(
                        // leaving two spaces for indentation.
                        "  {id} [label=\"{label}\" shape=box]\n  {links}\n",
                        id = node.id(),
                        label = node.dot_label(),
                        links = node.dot_links(),
                    )
                })
                .collect::<String>();
            if let Some(name) = name {
                Dot(format!(
                    "digraph {name} {{\n{graph_body}\n}}",
                    graph_body = graph_body,
                ))
            } else {
                Dot(format!(
                    "digraph {{\n{graph_body}\n}}",
                    graph_body = graph_body,
                ))
            }
        }
    }
}

mod simplify {
    use crate::ir::Graph;

    impl Graph {
        /// removes any unnecessary nodes from the graph.
        pub fn shrink(&mut self) {
            let mut bfs = self.bfs_mut();

            while let Some(mut node) = bfs.next() {
                // we do not visit the same node twice so we have to consume as mutch as possible.
                while node.as_mut().merge_tail_if(|tail| tail.incoming_len() == 1) {}
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::hir::*;
    use crate::lexer::*;
    use crate::parser::*;
    use crate::span::*;

    #[test]
    fn empty_main() {
        let code = SpanSource::new(br#"void main() { return ; }"#);
        let mut parser = Parser::new(
            tokens(code.source()).map(|s| s.map(|t| t.unwrap())),
            |_| unreachable!(),
        );
        let proot = parser.doc_elems().collect();
        let hirtree = HIRRoot::from_proot(proot).unwrap();

        let _graph = hirtree.functions.values().next().unwrap().destruct();
    }

    #[test]
    fn add() {
        let code = SpanSource::new(
            br#"void main() { return ; } int add(int a, int b) { int c; c = a + b; return c; }"#,
        );
        let mut parser = Parser::new(
            tokens(code.source()).map(|s| s.map(|t| t.unwrap())),
            |_| unreachable!(),
        );
        let proot = parser.doc_elems().collect();
        let hirtree = HIRRoot::from_proot(proot).unwrap();

        hirtree.functions.values().for_each(|func| {
            func.destruct();
        });
    }
}
