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
            Some(&Terminator::Fork { yes, no, cond, .. }) => Some(NeighbouringNodes::Conditional {
                cond,
                yes: Node {
                    block: yes,
                    _marker: Marker,
                },
                no: Node {
                    block: no,
                    _marker: Marker,
                },
            }),
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

    /// removes any unnecessary nodes from the graph.
    pub fn shrink(&mut self) {
        let mut bfs = self.bfs_mut();

        while let Some(mut node) = bfs.next() {
            // we do not visit the same node twice so we have to consume as mutch as possible.
            while node.as_mut().merge_tail_if(|tail| tail.incoming_len() == 1) {}
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
