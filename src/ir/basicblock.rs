use crate::ir::{cfg::Cfg, instruction::*};
use std::{cell::RefCell, collections::HashSet, rc::Rc};

pub type BasicBlockRef = Rc<RefCell<BasicBlock>>;

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

#[derive(Debug, Clone)]
pub enum Terminator {
    Fork {
        cond: Reg,
        yes: BasicBlockRef,
        no: BasicBlockRef,
    },
    Tail {
        block: BasicBlockRef,
    },
}

#[derive(Default)]
pub struct BasicBlockBuilder {
    instructions: Option<Vec<Instruction>>,
    terminator: Option<Terminator>,
}

impl BasicBlockBuilder {
    pub fn new() -> Self {
        Self {
            instructions: None,
            terminator: None,
        }
    }

    pub fn with_instructions(self, instructions: impl IntoIterator<Item = Instruction>) -> Self {
        Self {
            instructions: Some(instructions.into_iter().collect()),
            terminator: self.terminator,
        }
    }

    pub fn with_termiator(self, terminator: Terminator) -> Self {
        Self {
            instructions: self.instructions,
            terminator: Some(terminator),
        }
    }

    pub fn with_tail(self, tail: BasicBlockRef) -> Self {
        self.with_termiator(Terminator::Tail { block: tail })
    }

    pub fn with_fork(self, cond: Reg, yes: BasicBlockRef, no: BasicBlockRef) -> Self {
        self.with_termiator(Terminator::Fork { cond, yes, no })
    }

    pub fn build(self) -> BasicBlock {
        BasicBlock {
            instructions: self.instructions.unwrap_or(vec![]),
            terminator: self.terminator,
        }
    }

    pub fn build_refcount(self) -> BasicBlockRef {
        self.build().into_refcount()
    }

    pub fn build_cfg(self) -> Cfg {
        self.build_refcount().into()
    }
}

#[derive(Debug, Clone)]
pub struct BasicBlock {
    instructions: Vec<Instruction>,
    terminator: Option<Terminator>,
}

pub struct IRFunction {
    root: BasicBlockRef,
}

pub trait NodeVisit {
    fn visit(&mut self, node: Rc<RefCell<BasicBlock>>);
}

impl<F> NodeVisit for F
where
    F: FnMut(Rc<RefCell<BasicBlock>>),
{
    fn visit(&mut self, node: Rc<RefCell<BasicBlock>>) {
        (*self)(node)
    }
}

struct Graph(BasicBlockRef);

enum NextNode {
    Recurse(BasicBlockRef),
    Single(BasicBlockRef),
}

pub struct Node(BasicBlockRef);

pub struct Dfs {
    ignore: HashSet<*const RefCell<BasicBlock>>,
    stack: Vec<NextNode>,
}

impl Dfs {
    fn push(&mut self, node: NextNode) {
        match node {
            NextNode::Recurse(node) => {
                if self.ignore.insert(Rc::as_ptr(&node)) {
                    self.stack.push(NextNode::Recurse(node))
                }
            }
            // this is not a typo. we do not check if single nodes are ignored.
            // TODO: explain why
            NextNode::Single(node) => self.stack.push(NextNode::Single(node)),
        }
    }
}

impl Node {
    pub fn dfs(self) -> Dfs {
        Dfs {
            ignore: HashSet::from([Rc::as_ptr(&self.0)]),
            stack: vec![NextNode::Recurse(self.0)],
        }
    }
}

impl Iterator for Dfs {
    type Item = BasicBlockRef;
    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            Some(NextNode::Single(node)) => Some(node),
            Some(NextNode::Recurse(node)) => match node.borrow().terminator() {
                None => Some(Rc::clone(&node)),
                Some(Terminator::Fork { yes, no, .. }) => {
                    self.push(NextNode::Single(Rc::clone(&node)));
                    self.push(NextNode::Recurse(Rc::clone(yes)));
                    self.push(NextNode::Recurse(Rc::clone(no)));
                    self.next()
                }
                Some(Terminator::Tail { block }) => {
                    self.push(NextNode::Single(Rc::clone(&node)));
                    self.push(NextNode::Recurse(Rc::clone(block)));
                    self.next()
                }
            },
            None => None,
        }
    }
}

impl IRFunction {
    pub fn new(root: BasicBlockRef) -> Self {
        Self { root }
    }

    pub fn root(&self) -> Node {
        Node(Rc::clone(&self.root))
    }

    pub fn to_dot(&self) -> String {
        let mut dot_graph_internal = String::new();
        // TODO: this is for debuging delete this
        let mut counter = 0;
        self.root().dfs().for_each(|node: Rc<RefCell<BasicBlock>>| {
            let label = node
                .borrow()
                .instructions()
                .fold(String::new(), |acc, inst| format!("{}{:?}\\n", acc, inst));
            let terminator = node.borrow().terminator().as_ref().map_or_else(
                || "".to_string(),
                |term| match term {
                    Terminator::Fork { yes, no, .. } => {
                        format!(
                            "{:?} -> {{{:?} {:?}}}\n",
                            node.as_ptr() as usize,
                            yes.as_ptr() as usize,
                            no.as_ptr() as usize,
                        )
                    }
                    Terminator::Tail { block: tail } => {
                        format!(
                            "{:?} -> {:?}\n",
                            node.as_ptr() as usize,
                            tail.as_ptr() as usize
                        )
                    }
                },
            );
            counter += 1;
            dot_graph_internal.push_str(&format!(
                "{:?} [label=\"{}\\n{counter}\" shape=box]\n{}",
                node.as_ptr() as usize,
                label,
                terminator
            ));
        });
        format!("digraph {{\n{}}}", dot_graph_internal)
    }
}

impl BasicBlock {
    pub fn terminator(&self) -> &Option<Terminator> {
        &self.terminator
    }

    pub fn terminator_mut(&mut self) -> &mut Option<Terminator> {
        &mut self.terminator
    }

    pub fn take_terminator(&mut self) -> Option<Terminator> {
        self.terminator.take()
    }

    pub fn take_instructions(&mut self) -> Vec<Instruction> {
        std::mem::take(&mut self.instructions)
    }

    pub fn instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.instructions.iter()
    }

    pub fn into_refcount(self) -> BasicBlockRef {
        BasicBlockRef::new(self.into())
    }

    pub fn set_tail(&mut self, tail: BasicBlockRef) {
        self.terminator = Some(Terminator::Tail { block: tail });
    }

    pub fn set_terminator(&mut self, terminator: Option<Terminator>) {
        self.terminator = terminator;
    }

    pub fn extend_instructions(&mut self, instructions: impl IntoIterator<Item = Instruction>) {
        self.instructions.extend(instructions);
    }
}

impl Extend<Instruction> for BasicBlock {
    fn extend<T: IntoIterator<Item = Instruction>>(&mut self, iter: T) {
        self.instructions.extend(iter);
    }
}
