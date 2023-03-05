use crate::ir::basicblock::BasicBlockRef;

#[derive(Clone)]
pub struct Cfg {
    pub beg: BasicBlockRef,
    pub end: BasicBlockRef,
}

impl Cfg {
    pub fn new(beg: BasicBlockRef, end: BasicBlockRef) -> Self {
        Self { beg, end }
    }
    pub fn concat(self, end: Self) -> Self {
        self.end.borrow_mut().set_tail(end.beg);
        Self {
            beg: self.beg,
            end: end.end,
        }
    }
}

impl From<BasicBlockRef> for Cfg {
    fn from(value: BasicBlockRef) -> Self {
        Cfg {
            beg: value.clone(),
            end: value,
        }
    }
}
