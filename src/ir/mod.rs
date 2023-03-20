//! TODO: this module requries alot of refactoring.

mod instruction;
#[cfg(test)]
mod test;
pub use basicblock::*;
pub use dot::*;
pub use graph::*;
pub use instruction::*;

mod basicblock {
    pub use crate::ir::instruction::*;
    use std::{collections::HashSet, ptr::NonNull};

    #[derive(Default)]
    pub struct RegAllocator(u32);

    impl RegAllocator {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn alloc_ssa(&mut self) -> Reg {
            let reg = Reg::new(self.0);
            self.0 += 1;
            reg
        }

        pub fn allocated(self) -> u32 {
            self.0
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub(super) enum Terminator<'a> {
        Fork {
            cond: Reg,
            yes: NonNull<BasicBlock<'a>>,
            no: NonNull<BasicBlock<'a>>,
        },
        Tail {
            block: NonNull<BasicBlock<'a>>,
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

    #[allow(unused)]
    #[derive(Debug, Clone, Default)]
    pub struct BBMetaData {
        ty: Option<BBType>,
        is_root: bool,
    }

    impl BBMetaData {
        pub fn new(ty: Option<BBType>) -> Self {
            Self { ty, is_root: false }
        }

        pub fn build_block<'b>(self, insrtctions: &[Instruction<'b>]) -> BasicBlock<'b> {
            BasicBlock::new(self, insrtctions)
        }
    }

    #[derive(Debug, Clone, Default)]
    pub struct BasicBlock<'b> {
        incoming: HashSet<NonNull<BasicBlock<'b>>>,
        instructions: Vec<Instruction<'b>>,
        terminator: Option<Terminator<'b>>,
        metadata: BBMetaData,
    }

    impl<'b> BasicBlock<'b> {
        pub(super) fn terminator(&self) -> Option<&Terminator<'b>> {
            self.terminator.as_ref()
        }

        pub fn instructions(&self) -> &[Instruction<'b>] {
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

        pub fn instructions_mut(&mut self) -> &mut Vec<Instruction<'b>> {
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

        pub fn push(&mut self, instruction: Instruction<'b>) {
            self.instructions.push(instruction);
        }

        pub fn new(metadata: BBMetaData, instruction: &[Instruction<'b>]) -> Self {
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
        pub fn tail(&self) -> Option<&BasicBlock<'b>> {
            match self.terminator {
                Some(Terminator::Tail { block, .. }) => Some(unsafe { block.as_ref() }),
                _ => None,
            }
        }

        /// merges the tail block if it is a tail block. and `f(tail)` returns true.
        ///
        /// returns `true` if the tail was mereged
        pub fn merge_tail_if(&mut self, p: impl FnOnce(&BasicBlock<'b>) -> bool) -> bool {
            if let Some(tail) = self.tail() {
                if p(tail) {
                    self.instructions.extend(tail.instructions.clone());
                    self.bypass_tail();
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

    pub struct Graph<'a> {
        root: NonNull<BasicBlock<'a>>,
        _marker: Marker<BasicBlock<'a>>,
    }

    impl Drop for Graph<'_> {
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

    pub struct Node<'b> {
        block: NonNull<BasicBlock<'b>>,
        _marker: Marker<&'b BasicBlock<'b>>,
    }

    pub struct NodeMut<'b> {
        block: NonNull<BasicBlock<'b>>,
        _marker: Marker<&'b BasicBlock<'b>>,
    }

    impl<'b> AsRef<BasicBlock<'b>> for Node<'b> {
        fn as_ref(&self) -> &BasicBlock<'b> {
            unsafe { self.block.as_ref() }
        }
    }

    impl<'b> AsRef<BasicBlock<'b>> for NodeMut<'b> {
        fn as_ref(&self) -> &BasicBlock<'b> {
            unsafe { self.block.as_ref() }
        }
    }

    impl<'b> AsMut<BasicBlock<'b>> for NodeMut<'b> {
        fn as_mut(&mut self) -> &mut BasicBlock<'b> {
            unsafe { self.block.as_mut() }
        }
    }

    impl<'b> NodeMut<'b> {
        pub fn instructions_mut(&mut self) -> &mut Vec<Instruction<'b>> {
            self.as_mut().instructions_mut()
        }

        fn terminator(&self) -> Option<&Terminator<'b>> {
            self.as_ref().terminator()
        }

        pub fn tail(&self) -> Option<Node<'b>> {
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

        pub fn as_node(&self) -> Node<'b> {
            Node {
                block: self.block,
                _marker: Marker,
            }
        }

        pub fn is_leaf(&self) -> bool {
            self.as_ref().terminator().is_none()
        }

        pub fn merge_tail_if(&mut self, p: impl FnOnce(&BasicBlock<'b>) -> bool) -> bool {
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
    pub enum NeighbouringNodes<'b> {
        Unconditional {
            next: Node<'b>,
        },
        Conditional {
            cond: Reg,
            yes: Node<'b>,
            no: Node<'b>,
        },
    }

    impl<'b> Node<'b> {
        fn terminator(&self) -> Option<&Terminator<'b>> {
            self.as_ref().terminator()
        }

        pub fn is_leaf(&self) -> bool {
            self.terminator().is_none()
        }

        pub fn instructions(&self) -> &'b [Instruction<'b>] {
            unsafe { self.block.as_ref().instructions() }
        }

        pub fn neighbours(&self) -> Option<NeighbouringNodes<'b>> {
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

        /// returns a unique identifier for the basicblock. The id is unique for the hole program.
        /// (**ignoring ids for dropped nodes**).
        pub fn id(&self) -> NodeId {
            NodeId(self.as_ref() as *const _ as usize)
        }
    }

    enum DfsNode<'a> {
        Recurse(NonNull<BasicBlock<'a>>),
        Shallow(NonNull<BasicBlock<'a>>),
    }

    pub struct Dfs<'b> {
        visited: HashSet<NonNull<BasicBlock<'b>>>,
        stack: Vec<DfsNode<'b>>,
        _marker: Marker<&'b BasicBlock<'b>>,
    }

    pub struct DfsMut<'b> {
        visited: HashSet<NonNull<BasicBlock<'b>>>,
        stack: Vec<DfsNode<'b>>,
        _marker: Marker<&'b mut BasicBlock<'b>>,
    }

    pub struct PsudoInOrder<'b> {
        stack: Vec<NonNull<BasicBlock<'b>>>,
        /// we have to keep the visited node because there are cycles in a typical cfg.
        visited: HashSet<NonNull<BasicBlock<'b>>>,
        _marker: Marker<&'b BasicBlock<'b>>,
    }

    #[derive(Debug)]
    struct BfsBookKeeper<'b> {
        visited: HashSet<NonNull<BasicBlock<'b>>>,
        queue: VecDeque<NonNull<BasicBlock<'b>>>,
    }

    /// a bfs iterator that allows mutating nodes. the nodes can mutate two things the instructions
    /// list of a block and the outgoing edges in the current `BfsMutNode`.
    pub struct BfsMut<'b> {
        book_keeper: BfsBookKeeper<'b>,
        _marker: Marker<&'b mut BasicBlock<'b>>,
    }

    impl<'b> BfsBookKeeper<'b> {
        /// conditionally adds a node to the queue
        fn enqueue(&mut self, node: NonNull<BasicBlock<'b>>) {
            if self.visited.insert(node) {
                self.queue.push_back(node);
            }
        }

        /// returns the next node to be visited (if it exists).
        fn dequeue(&mut self) -> Option<NonNull<BasicBlock<'b>>> {
            self.queue.pop_front()
        }

        fn clear(&mut self) {
            assert!(self.queue.is_empty());
            self.visited.clear();
        }
    }

    impl<'b> BfsMut<'b> {
        #[allow(clippy::should_implement_trait)]
        pub fn next<'s>(&'s mut self) -> Option<BfsMutNode<'s, 'b>> {
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

    impl<'b> DfsMut<'b> {
        #[allow(clippy::should_implement_trait)]
        pub fn next(&mut self) -> Option<NodeMut<'b>> {
            if let Some(top) = self.stack.pop() {
                match top {
                    DfsNode::Shallow(node) => Some(NodeMut {
                        block: node,
                        _marker: Marker,
                    }),
                    DfsNode::Recurse(node) => match unsafe { node.as_ref() }.terminator() {
                        None => Some(NodeMut {
                            block: node,
                            _marker: Marker,
                        }),
                        Some(Terminator::Tail { block, .. }) => {
                            self.stack.push(DfsNode::Shallow(node));
                            if !self.visited.contains(block) {
                                self.stack.push(DfsNode::Recurse(*block));
                                self.visited.insert(*block);
                            }
                            self.next()
                        }
                        Some(Terminator::Fork { yes, no, .. }) => {
                            self.stack.push(DfsNode::Shallow(node));
                            if !self.visited.contains(yes) {
                                self.stack.push(DfsNode::Recurse(*yes));
                                self.visited.insert(*yes);
                            }
                            if !self.visited.contains(no) {
                                self.stack.push(DfsNode::Recurse(*no));
                                self.visited.insert(*no);
                            }
                            self.next()
                        }
                    },
                }
            } else {
                None
            }
        }
    }

    pub struct BfsMutNode<'k, 'b> {
        /// a refrence to the bfs iterator so that we can update the queue when the node is getting
        /// destructed.
        book_keeper: &'k mut BfsBookKeeper<'b>,
        node: NodeMut<'b>,
    }

    impl<'b> AsRef<NodeMut<'b>> for BfsMutNode<'_, 'b> {
        fn as_ref(&self) -> &NodeMut<'b> {
            &self.node
        }
    }

    impl<'b> AsMut<NodeMut<'b>> for BfsMutNode<'_, 'b> {
        fn as_mut(&mut self) -> &mut NodeMut<'b> {
            &mut self.node
        }
    }

    impl<'b> Drop for BfsMutNode<'_, 'b> {
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

    impl<'b> Dfs<'b> {
        fn new(root: &BasicBlock<'b>) -> Self {
            Self {
                visited: HashSet::from([NonNull::new(root as *const _ as *mut _).unwrap()]),
                stack: vec![DfsNode::Recurse(
                    NonNull::new(root as *const _ as *mut _).unwrap(),
                )],
                _marker: Marker,
            }
        }
    }

    impl<'b> Iterator for Dfs<'b> {
        type Item = Node<'b>;
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(top) = self.stack.pop() {
                match top {
                    DfsNode::Shallow(node) => Some(Node {
                        block: node,
                        _marker: Marker,
                    }),
                    DfsNode::Recurse(node) => match unsafe { node.as_ref() }.terminator() {
                        None => Some(Node {
                            block: node,
                            _marker: Marker,
                        }),
                        Some(Terminator::Tail { block, .. }) => {
                            self.stack.push(DfsNode::Shallow(node));
                            // PERF: it is quite hard for the compiler to optimize this
                            if !self.visited.contains(block) {
                                self.stack.push(DfsNode::Recurse(*block));
                                self.visited.insert(*block);
                            }
                            self.next()
                        }
                        Some(Terminator::Fork { yes, no, .. }) => {
                            self.stack.push(DfsNode::Shallow(node));
                            if !self.visited.contains(yes) {
                                self.stack.push(DfsNode::Recurse(*yes));
                                self.visited.insert(*yes);
                            }
                            if !self.visited.contains(no) {
                                self.stack.push(DfsNode::Recurse(*no));
                                self.visited.insert(*no);
                            }
                            self.next()
                        }
                    },
                }
            } else {
                None
            }
        }
    }

    impl<'b> Iterator for PsudoInOrder<'b> {
        type Item = Node<'b>;
        fn next(&mut self) -> Option<Self::Item> {
            if let Some(node) = self.stack.pop() {
                self.visited.insert(node);
                let node = Node {
                    block: node,
                    _marker: Marker,
                };
                match node.terminator() {
                    None => Some(node),
                    Some(Terminator::Tail { block, .. }) => {
                        if !self.visited.contains(block) {
                            self.stack.push(*block);
                        }
                        Some(node)
                    }
                    Some(Terminator::Fork { yes, no, .. }) => {
                        if !self.visited.contains(no) {
                            self.stack.push(*no);
                        }
                        if !self.visited.contains(yes) {
                            self.stack.push(*yes);
                        }
                        Some(node)
                    }
                }
            } else {
                self.visited.clear();
                None
            }
        }
    }

    impl<'b> Graph<'b> {
        pub fn new(root: NonNull<BasicBlock<'b>>) -> Self {
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

        pub fn root(&self) -> &BasicBlock<'b> {
            unsafe { self.root.as_ref() }
        }

        pub fn root_mut(&mut self) -> &mut BasicBlock<'b> {
            unsafe { self.root.as_mut() }
        }

        pub fn dfs(&self) -> Dfs<'b> {
            Dfs::new(self.root())
        }

        pub fn dfs_mut(&mut self) -> DfsMut<'b> {
            DfsMut {
                visited: HashSet::from([self.root]),
                stack: vec![DfsNode::Recurse(self.root)],
                _marker: Marker,
            }
        }

        pub fn psudo_inorder(&self) -> PsudoInOrder<'b> {
            PsudoInOrder {
                stack: vec![self.root],
                visited: HashSet::default(),
                _marker: Marker,
            }
        }

        pub fn bfs_mut(&mut self) -> BfsMut<'b> {
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

    pub struct Function<'b> {
        pub(super) graph: Graph<'b>,
        name: &'b str,
        allocated_regs: u32,
        args: Vec<Symbol<'b>>,
    }

    impl<'b> AsRef<Graph<'b>> for Function<'b> {
        fn as_ref(&self) -> &Graph<'b> {
            self.graph()
        }
    }

    impl<'b> AsMut<Graph<'b>> for Function<'b> {
        fn as_mut(&mut self) -> &mut Graph<'b> {
            self.graph_mut()
        }
    }

    impl<'b> Function<'b> {
        pub fn new(
            name: &'b str,
            graph: Graph<'b>,
            allocated_regs: u32,
            args: impl IntoIterator<Item = Symbol<'b>>,
        ) -> Self {
            Self {
                name,
                args: args.into_iter().collect(),
                allocated_regs,
                graph,
            }
        }

        pub fn allocate_reg(&mut self) -> Reg {
            self.allocated_regs += 1;
            Reg::new(self.allocated_regs - 1)
        }

        pub fn allocated_regs(&self) -> u32 {
            self.allocated_regs
        }

        pub fn args(&self) -> &[Symbol] {
            &self.args
        }

        pub fn to_dot(&self) -> Dot {
            let dot = self.as_ref().to_dot(Some(self.name()));
            println!("{}", dot);
            dot
        }

        pub fn name(&self) -> &'b str {
            self.name
        }

        pub fn graph(&self) -> &Graph<'b> {
            &self.graph
        }

        pub fn graph_mut(&mut self) -> &mut Graph<'b> {
            &mut self.graph
        }
    }

    pub struct Program<'b> {
        pub(super) functions: Vec<Function<'b>>,
        pub(super) globals: Vec<(&'b str, usize)>,
    }

    impl<'b> Program<'b> {
        pub fn functions(&self) -> &[Function<'b>] {
            &self.functions
        }

        pub fn globals(&self) -> &[(&'b str, usize)] {
            &self.globals
        }

        pub fn new<F: Into<Function<'b>>>(
            globals: impl IntoIterator<Item = (&'b str, usize)>,
            functions: impl IntoIterator<Item = F>,
        ) -> Self {
            Self {
                functions: functions.into_iter().map(|f| f.into()).collect(),
                globals: globals.into_iter().collect(),
            }
        }
    }
}

mod ccfg {
    use crate::ir::{BBMetaData, BBType::*, BasicBlock, Graph, Reg};
    use std::{marker::PhantomData as Marker, ptr::NonNull};

    pub struct CCfg<'b> {
        beg: NonNull<BasicBlock<'b>>,
        end: Option<NonNull<BasicBlock<'b>>>,
        _marker: Marker<&'b BasicBlock<'b>>,
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
}

mod destruct {
    use std::collections::HashSet;
    use std::{cell::UnsafeCell, num::NonZeroU16};

    use super::{ccfg::*, IRExternArg, Loc, Program};
    use crate::{
        hir::*,
        ir::{BBMetaData, BasicBlock, Function, Immediate, Instruction, Reg, RegAllocator, Symbol},
    };

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
                        CCfg::new(Box::new(BasicBlock::new(
                            BBMetaData::new(None),
                            &[Instruction::new_load(
                                res,
                                Immediate::Int(size.get() as i64),
                            )],
                        ))),
                        res,
                    )
                }
                HIRExpr::Loc(loc) => {
                    let (mut ccfg, loc) = loc.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let assign = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_load(res, loc)],
                    )));
                    ccfg.append(assign);
                    (ccfg, res)
                }
                HIRExpr::Arith { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let equ = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_arith(res, lhs, *op, rhs)],
                    )));
                    ccfg_lhs.append(ccfg_rhs).append(equ);
                    (ccfg_lhs, res)
                }

                HIRExpr::Rel { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let equ = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_rel(res, lhs, *op, rhs)],
                    )));
                    ccfg_lhs.append(ccfg_rhs).append(equ);
                    (ccfg_lhs, res)
                }
                Eq { op, lhs, rhs } => {
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let equ = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_eq(res, lhs, *op, rhs)],
                    )));
                    ccfg_lhs.append(ccfg_rhs).append(equ);
                    (ccfg_lhs, res)
                }
                Neg(e) => {
                    let (mut ccfg, res_neg) = e.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let bb = BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_neg(res, res_neg)],
                    );
                    ccfg.append(CCfg::new(Box::new(bb)));
                    (ccfg, res)
                }
                Not(e) => {
                    let (mut ccfg, res_not) = e.destruct(reg_allocator, mangler, func_name);
                    let res = reg_allocator.alloc_ssa();
                    let bb = BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_not(res, res_not)],
                    );
                    ccfg.append(CCfg::new(Box::new(bb)));
                    (ccfg, res)
                }
                Ter { cond, yes, no } => {
                    let res = mangler.cook_var();
                    let (mut ccfg_cond, cond) = cond.destruct(reg_allocator, mangler, func_name);
                    ccfg_cond.append(CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::AllocScalar { name: res }],
                    ))));
                    let (mut ccfg_yes, yes) = yes.destruct(reg_allocator, mangler, func_name);
                    ccfg_yes.append(CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(res, yes)],
                    ))));
                    let (mut ccfg_no, no) = no.destruct(reg_allocator, mangler, func_name);
                    ccfg_no.append(CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(res, no)],
                    ))));
                    ccfg_cond.append_cond(cond, ccfg_yes, ccfg_no);
                    let res_reg = reg_allocator.alloc_ssa();
                    let assign = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_load(res_reg, res)],
                    )));
                    ccfg_cond.append(assign);
                    (ccfg_cond, res_reg)
                }
                Call(call) => call.destruct_ret(reg_allocator, mangler, func_name),
                Cond {
                    lhs,
                    rhs,
                    op: CondOp::Or,
                } => {
                    let sc = mangler.cook_var();
                    let scbb = Box::new(BasicBlock::new(
                        BBMetaData::default(),
                        &[Instruction::AllocScalar { name: sc }],
                    ));
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (mut ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let set_true = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(sc, true)],
                    )));
                    let set_false = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(sc, false)],
                    )));
                    ccfg_rhs.append_cond(rhs, set_true, set_false);
                    let set_true = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(sc, true)],
                    )));
                    ccfg_lhs.append_cond(lhs, set_true, ccfg_rhs);
                    let mut beg = CCfg::new(scbb);
                    beg.append(ccfg_lhs);
                    let res = reg_allocator.alloc_ssa();
                    let load = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_load(res, sc)],
                    )));
                    beg.append(load);
                    (beg, res)
                }
                Cond {
                    op: CondOp::And,
                    lhs,
                    rhs,
                } => {
                    let sc = mangler.cook_var();
                    let scbb = Box::new(BasicBlock::new(
                        BBMetaData::default(),
                        &[Instruction::AllocScalar { name: sc }],
                    ));
                    let (mut ccfg_lhs, lhs) = lhs.destruct(reg_allocator, mangler, func_name);
                    let (mut ccfg_rhs, rhs) = rhs.destruct(reg_allocator, mangler, func_name);
                    let set_false = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(sc, false)],
                    )));
                    let set_true = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(sc, true)],
                    )));
                    ccfg_rhs.append_cond(rhs, set_true, set_false);
                    let set_false = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_store(sc, false)],
                    )));
                    ccfg_lhs.append_cond(lhs, ccfg_rhs, set_false);
                    let mut beg = CCfg::new(scbb);
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
                    let call = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_ret_call(res, *name, args)],
                    )));
                    ccfg.append(call);
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
                    let call = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[Instruction::new_extern_call(res, *name, args)],
                    )));
                    ccfg.append(call);
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
                    let mut bound_check = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[
                            Instruction::new_rel(
                                less_than_zero,
                                index,
                                RelOp::Less,
                                Immediate::Int(0),
                            ),
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
                        ],
                    )));
                    let on_bound_fail = CCfg::new(Box::new(BasicBlock::new(
                        BBMetaData::new(None),
                        &[
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
                        ],
                    )));
                    bound_check.append_cond(bound_cond, on_bound_fail, CCfg::new_empty());
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
            let assign = CCfg::new(Box::new(BasicBlock::new(
                BBMetaData::new(None),
                &match self.rhs {
                    AssignOp::AddAssign(_) => [Instruction::new_arith(lhs, lhs, ArithOp::Add, rhs)],
                    AssignOp::SubAssign(_) => [Instruction::new_arith(lhs, lhs, ArithOp::Sub, rhs)],
                    AssignOp::Assign(_) => [Instruction::new_store(lhs, rhs)],
                },
            )));
            ccfg_loc.append(ccfg).append(assign);
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
            let mut ccfg = CCfg::new(Box::new(BasicBlock::new(
                BBMetaData::default(),
                &stack_allocs,
            )));
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
            let empty = vec![];
            let mut ccfg = CCfg::new(Box::new(BasicBlock::new(BBMetaData::default(), &empty)));
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

mod dot {
    use std::{
        fmt::Display,
        io::{self, Write},
        path::Path,
        process::{Command, ExitStatus, Stdio},
    };

    use crate::ir::{Graph, NeighbouringNodes, Node};

    pub struct Dot(String);

    impl Display for Dot {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

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
                .instructions()
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

    impl Graph<'_> {
        /// converts the graph to a dot graph.
        pub fn to_dot<T: Display>(&self, name: Option<T>) -> Dot {
            let graph_body = self
                .dfs()
                .map(|node| {
                    println!("address: {:?}", node.as_ref() as *const _);
                    println!("{:?}", node.as_ref());
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

    impl Graph<'_> {
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
