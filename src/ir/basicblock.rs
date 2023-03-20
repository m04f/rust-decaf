pub use crate::ir::instruction::*;
use std::{collections::HashSet, ptr::NonNull};

#[derive(Default)]
pub(super) struct RegAllocator(u32);

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

impl<'b, const N: usize> From<&[Instruction<'b>; N]> for BasicBlock<'b> {
    fn from(value: &[Instruction<'b>; N]) -> Self {
        BasicBlock::new(BBMetaData::default(), value)
    }
}

impl<'b> From<&[Instruction<'b>]> for BasicBlock<'b> {
    fn from(value: &[Instruction<'b>]) -> Self {
        BasicBlock::new(BBMetaData::default(), value)
    }
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
