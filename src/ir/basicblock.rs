use crate::ir::{arena::*, instruction::*};

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

pub struct BasicBlock {
    refs: usize,
    instructions: Vec<Instruction>,
    terminator: Option<Terminator>,
}

impl Default for BasicBlock {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicBlock {
    pub fn new() -> Self {
        Self {
            refs: 0,
            instructions: Vec::new(),
            terminator: None,
        }
    }

    pub fn terminator(&self) -> Option<&Terminator> {
        self.terminator.as_ref()
    }

    pub fn instructions(&self) -> impl Iterator<Item = &Instruction> {
        self.instructions.iter()
    }

    pub fn push(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    pub fn set_tail(&mut self, tail: BasicBlockRef, arena: &Arena) {
        arena.get_block_mut(tail).refs += 1;
        self.terminator = Some(Terminator::Tail { block: tail });
    }

    pub fn set_fork(&mut self, cond: Reg, yes: BasicBlockRef, no: BasicBlockRef, arena: &Arena) {
        arena.get_block_mut(yes).refs += 1;
        arena.get_block_mut(no).refs += 1;
        self.terminator = Some(Terminator::Fork { cond, yes, no });
    }

    pub fn simplify(&mut self, arena: &Arena) -> bool {
        match self.terminator.as_ref() {
            Some(Terminator::Tail { block }) => {
                // really...
                let block = *block;
                if arena.get_block(block).refs == 1 {
                    self.instructions
                        .extend(arena.get_block(block).instructions.iter().cloned());
                    self.terminator = arena.get_block(block).terminator().cloned();
                    arena.drop_block(block);
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }
}

impl Extend<Instruction> for BasicBlock {
    fn extend<T: IntoIterator<Item = Instruction>>(&mut self, iter: T) {
        self.instructions.extend(iter);
    }
}
