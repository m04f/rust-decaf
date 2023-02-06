use crate::dequeue::Dequeue;

pub struct LookAhead<Elem, I: Iterator<Item = Elem>, const MAX_LOOK_AHEAD: usize> {
    iter: I,
    looked: Dequeue<Elem, MAX_LOOK_AHEAD>,
}

impl<Elem, I: Iterator<Item = Elem>, const MAX_LOOK_AHEAD: usize>
    LookAhead<Elem, I, MAX_LOOK_AHEAD>
{
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            looked: Dequeue::new(),
        }
    }

    pub fn peek(&mut self) -> Option<&I::Item> {
        if self.looked.is_empty() {
            self.looked.enqueue(self.iter.next()?);
        }
        self.looked.front()
    }

    pub fn look_ahead<const VALUE: usize>(&mut self) -> Option<&I::Item>
    {
        assert!(VALUE < MAX_LOOK_AHEAD);
        if self.looked.len() <= VALUE {
            while self.looked.len() <= VALUE {
                self.looked.enqueue(self.iter.next()?);
            }
        }
        self.looked.get(VALUE)
    }
}

impl<Elem, I: Iterator<Item = Elem>, const MAX_LOOK_AHEAD: usize> Iterator
    for LookAhead<Elem, I, MAX_LOOK_AHEAD>
{
    type Item = Elem;

    fn next(&mut self) -> Option<Self::Item> {
        if self.looked.is_empty() {
            self.iter.next()
        } else {
            self.looked.dequeue()
        }
    }
}
