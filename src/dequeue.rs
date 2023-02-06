//! a dequeue of constant size

use std::marker::Destruct;
pub struct Dequeue<T, const SIZE: usize> {
    data: [T; SIZE],
    head: usize,
    len: usize,
}

impl<T, const SIZE: usize> From<[T; SIZE]> for Dequeue<T, SIZE> {
    fn from(value: [T; SIZE]) -> Self {
        Dequeue {
            data: value,
            head: 0,
            len: 0,
        }
    }
}

impl<T: Sized, const SIZE: usize> Dequeue<T, SIZE> {
    pub const fn new() -> Self {
        use std::mem;
        assert!(SIZE > 0);
        Dequeue {
            data: unsafe { mem::MaybeUninit::uninit().assume_init() },
            head: 0,
            len: 0,
        }
    }

    /// panics if there is no room for the new element
    pub const fn enqueue(&mut self, elem: T)
    where
        T: ~const Destruct,
    {
        assert!(!self.is_full());
        self.data[(self.head + self.len) % SIZE] = elem;
        self.len += 1;
    }

    pub fn front(&self) -> Option<&T> {
        (!self.is_empty()).then(|| &self.data[self.head])
    }

    pub fn dequeue(&mut self) -> Option<T> {
        use std::mem;
        (!self.is_empty()).then_some({
            let elem = mem::replace(&mut self.data[self.head], unsafe {
                mem::MaybeUninit::uninit().assume_init()
            });
            self.head = (self.head + 1) % SIZE;
            self.len -= 1;
            elem
        })
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn max_size(&self) -> usize {
        SIZE
    }

    pub const fn is_full(&self) -> bool {
        self.len() == self.max_size()
    }

    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        (self.len() > index).then(|| &self.data[(self.head + index) % SIZE])
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    #[should_panic]
    fn empty_dequeue() {
        Dequeue::<u8, 0>::new();
    }

    #[test]
    #[should_panic]
    fn overflow_dequeue() {
        let mut dequeue = Dequeue::<u8, 1>::new();
        dequeue.enqueue(0);
        dequeue.enqueue(1);
    }

    #[test]
    #[should_panic]
    fn dequeue_empty() {
        let mut dequeue = Dequeue::<u8, 1>::new();
        dequeue.dequeue().unwrap();
    }

    /// tests if the enqueue panics before it's limit
    #[test]
    fn enqueue_only() {
        let mut dequeue = Dequeue::<_, 6>::new();
        for i in 0..6 {
            dequeue.enqueue(i);
        }
    }

    #[test]
    fn enqueue_dequeue() {
        let mut dequeue = Dequeue::<_, 6>::new();
        dequeue.enqueue(0);
        dequeue.enqueue(1);
        dequeue.enqueue(2);
        assert_eq!(dequeue.dequeue().unwrap(), 0);
        assert_eq!(dequeue.dequeue().unwrap(), 1);
        // fill the queue
        dequeue.enqueue(3);
        dequeue.enqueue(4);
        dequeue.enqueue(5);
        dequeue.enqueue(6);
        dequeue.enqueue(7);
        for i in 2..=7 {
            assert_eq!(dequeue.dequeue().unwrap(), i);
        }
    }
}
