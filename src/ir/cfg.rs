use crate::ir::arena::*;

pub struct Cfg {
    pub arena: Arena,
    pub name: String,
    pub beg: BasicBlockRef,
    pub end: BasicBlockRef,
}

pub struct SubCfg {
    pub beg: BasicBlockRef,
    pub end: BasicBlockRef,
}

impl SubCfg {
    pub fn new(beg: BasicBlockRef, end: BasicBlockRef) -> Self {
        Self { beg, end }
    }
    pub fn concat(self, end: Self, arena: &Arena) -> Self {
        assert!(arena.get_block(self.end).terminator().is_none());
        arena.get_block_mut(self.end).set_tail(end.beg, arena);
        Self::new(self.beg, end.end)
    }
}

impl From<BasicBlockRef> for SubCfg {
    fn from(value: BasicBlockRef) -> Self {
        SubCfg {
            beg: value,
            end: value,
        }
    }
}

impl Cfg {
    pub fn to_dot(&self) -> String {
        use crate::ir::basicblock::*;
        let basicblock_dot = |block: BasicBlockRef| {
            let label = self
                .arena
                .get_block(block)
                .instructions()
                .fold(String::new(), |acc, inst| format!("{}{:?}\\l", acc, inst));
            let terminator = self.arena.get_block(block).terminator().map_or_else(
                || "".to_string(),
                |term| match term {
                    Terminator::Fork { yes, no, .. } => {
                        format!("{} -> {{{} {}}}", block.number(), yes.number(), no.number())
                    }
                    Terminator::Tail { block: tail } => {
                        format!("{} -> {}", block.number(), tail.number())
                    }
                },
            );
            format!("{} [label=\"{}\"]\n{}", block.number(), label, terminator)
        };

        format!(
            "digraph {} {{ \n{}\n}}",
            self.name,
            self.arena
                .blocks()
                .map(basicblock_dot)
                .fold(String::new(), |acc, block| format!("{}{}\n", acc, block))
        )
    }
}
