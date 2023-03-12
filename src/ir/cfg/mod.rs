mod construction;

pub use basicblock::{BBMetaData, BBType, BasicBlock};
pub use graph::{BfsMut, BfsMutNode, Cfg, Dfs, DfsMut, Dot, Node, NodeId, NodeMut};

use basicblock::Terminator;

mod basicblock {
    use crate::ir::instruction::{Instruction, Reg};

    use std::{collections::HashSet, ptr::NonNull};

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
        pub(super) fn from_instruction(instruction: Instruction) -> Self {
            Self {
                incoming: HashSet::new(),
                instructions: vec![instruction],
                terminator: None,
                metadata: BBMetaData::default(),
            }
        }

        pub(super) unsafe fn terminator(&self) -> Option<&Terminator> {
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

        unsafe fn incoming(&self) -> &HashSet<NonNull<BasicBlock>> {
            &self.incoming
        }

        pub fn instructions_mut(&mut self) -> &mut Vec<Instruction> {
            &mut self.instructions
        }

        pub fn incoming_len(&self) -> usize {
            self.incoming.len()
        }

        pub fn new(metadata: BBMetaData, instruction: &[Instruction]) -> Self {
            Self {
                incoming: HashSet::new(),
                instructions: instruction.to_vec(),
                terminator: None,
                metadata,
            }
        }

        pub(super) unsafe fn link_unconditional(&mut self, mut next: NonNull<Self>) {
            assert!(self.terminator.is_none());
            unsafe {
                self.terminator = Some(Terminator::Tail { block: next });
                assert!(next
                    .as_mut()
                    .incoming
                    .insert(NonNull::new(self as *mut Self).unwrap()));
            }
        }

        pub(super) unsafe fn link_conditional(
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
            match unsafe { self.terminator() } {
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
            if let Some(&Terminator::Tail { block }) = unsafe { self.terminator() } {
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
    pub use dot::Dot;

    use super::{BasicBlock, Terminator};

    use crate::ir::{Instruction, Reg};

    use std::{
        collections::{HashSet, VecDeque},
        fmt::Display,
        marker::PhantomData as Marker,
        ptr::NonNull,
    };

    pub struct Cfg {
        root: NonNull<BasicBlock>,
        _marker: Marker<BasicBlock>,
    }

    impl Drop for Cfg {
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
            unsafe { self.as_ref().terminator() }
        }

        pub fn tail(&self) -> Option<Node> {
            unsafe { self.as_ref().terminator() }.and_then(|t| match t {
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
            unsafe { self.as_ref().terminator() }.is_none()
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
            unsafe { self.as_ref().terminator() }
        }

        pub fn is_leaf(&self) -> bool {
            self.terminator().is_none()
        }

        pub fn neighbours(&self) -> Option<NeighbouringNodes<'a>> {
            match unsafe { self.as_ref().terminator() } {
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
                    DfsNode::Recurse(node) => match unsafe { node.as_ref().terminator() } {
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
                    DfsNode::Recurse(node) => match unsafe { node.as_ref().terminator() } {
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

    impl Cfg {
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
    mod dot {
        use std::{
            fmt::Display,
            io::{self, Write},
            path::Path,
            process::{Command, ExitStatus, Stdio},
        };

        use super::{Cfg, NeighbouringNodes, Node};

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

        impl Cfg {
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
}
