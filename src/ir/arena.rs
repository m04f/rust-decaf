use crate::ir::{basicblock::*, instruction::*};
use std::cell::{Ref, RefCell, RefMut};

pub struct Arena {
    arena: Vec<RefCell<BasicBlock>>,
    holes: Vec<BasicBlockRef>,
    registers_allocated: usize,
}

#[derive(Debug, Copy, Clone)]
pub struct BasicBlockRef(usize);

impl BasicBlockRef {
    pub fn number(&self) -> usize {
        self.0
    }
}

impl Arena {
    pub fn blocks(&self) -> impl Iterator<Item = BasicBlockRef> {
        (0..self.arena.len()).map(BasicBlockRef)
    }

    pub fn new() -> Self {
        Self {
            arena: Vec::new(),
            holes: Vec::new(),
            registers_allocated: 0,
        }
    }

    pub fn drop_block(&self, block_ref: BasicBlockRef) {
        todo!()
    }

    pub fn alloc_block(&mut self) -> BasicBlockRef {
        self.arena.push(RefCell::new(BasicBlock::new()));
        BasicBlockRef(self.arena.len() - 1)
    }

    pub fn get_block(&self, r#ref: BasicBlockRef) -> Ref<BasicBlock> {
        self.arena[r#ref.0].borrow()
    }

    pub fn get_block_mut(&self, r#ref: BasicBlockRef) -> RefMut<BasicBlock> {
        self.arena[r#ref.0].borrow_mut()
    }

    pub fn alloc_reg(&mut self) -> Reg {
        self.registers_allocated += 1;
        Reg::new(self.registers_allocated - 1)
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}
