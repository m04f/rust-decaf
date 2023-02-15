use core::ops::{Range, RangeFrom, RangeTo};
use std::fmt::Debug;
use std::ops::RangeFull;
use std::{iter::Copied, ops::Index, slice};

#[derive(PartialEq, Eq, Debug)]
pub struct Spanned<'a, T> {
    pub span: Span<'a>,
    pub data: T,
}

impl<T> Clone for Spanned<'_, T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            span: self.span,
            data: self.data.clone(),
        }
    }
}

impl<T> Copy for Spanned<'_, T> where T: Copy + Clone {}

impl<'a, T> Spanned<'a, T> {
    fn new(span: Span<'a>, data: T) -> Self {
        Self { span, data }
    }

    pub const fn span(&self) -> Span<'a> {
        self.span
    }

    pub fn get(&self) -> &T {
        &self.data
    }

    pub const fn copied(&self) -> T
    where
        T: Copy,
    {
        self.data
    }

    pub fn map<O, F: Fn(T) -> O>(self, f: F) -> Spanned<'a, O> {
        Spanned::new(self.span, f(self.data))
    }

    pub const fn fragment(&self) -> &'a [u8] {
        self.span.source()
    }

    pub const fn line(&self) -> u32 {
        self.span().line()
    }

    pub const fn column(&self) -> u32 {
        self.span().column()
    }

    pub const fn position(&self) -> (u32, u32) {
        self.span().position()
    }

    pub fn into_parts(self) -> (T, Span<'a>) {
        (self.data, self.span)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct Span<'a> {
    /// the part of the document spanned.
    source: &'a [u8],
    /// one indexed line number
    line: u32,
    /// one indexed column number
    column: u32,
}

impl<'a> Debug for Span<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        String::from_utf8_lossy(&self.source().to_vec()).fmt(f)
    }
}

impl<'a> Span<'a> {
    pub const fn new(source: &'a [u8]) -> Self {
        Self {
            source,
            line: 1,
            column: 1,
        }
    }

    pub const fn from_position(source: &'a [u8], (line, column): (u32, u32)) -> Self {
        Self {
            source,
            line,
            column,
        }
    }

    pub const fn position(&self) -> (u32, u32) {
        (self.line(), self.column())
    }

    pub const fn source(&self) -> &'a [u8] {
        self.source
    }

    pub fn bytes(&self) -> Copied<slice::Iter<u8>> {
        self.source().iter().copied()
    }

    pub const fn line(&self) -> u32 {
        self.line
    }

    pub const fn column(&self) -> u32 {
        self.column
    }

    pub const fn len(&self) -> usize {
        self.source().len()
    }

    /// returns the beginning of the first match to the given pattern
    /// NOTE: this is painfully slow
    pub fn find(&self, pat: &[u8]) -> Option<usize> {
        (0..self.len()).find(|&i| self[i..].starts_with(pat))
    }

    /// smilar to Slice::split_at
    /// | The first will contain all indices from `[0, mid)` (excluding
    /// | the index `mid` itself) and the second will contain all
    /// | indices from `[mid, len)` (excluding the index `len` itself).
    pub fn split_at(self, index: usize) -> (Self, Self) {
        let (left, right) = self.source().split_at(index);
        // left has the same position as the original span
        let left = Self::from_position(left, (self.line(), self.column()));
        let right_column = left
            .bytes()
            .rev()
            .position(|b| b == b'\n')
            // doing that silly -1 thing to avoid using clausures for mapping the value returned by
            // position
            .unwrap_or(left.len() + left.column() as usize - 1) as u32
            + 1;
        let right_line = left.line() + left.bytes().filter(|&b| b == b'\n').count() as u32;
        let right = Self::from_position(right, (right_line, right_column));
        (left, right)
    }

    pub fn starts_with(&self, pat: &[u8]) -> bool {
        self.source().starts_with(pat)
    }

    pub fn ends_with(&self, pat: &[u8]) -> bool {
        self.source().ends_with(pat)
    }

    pub fn take_while<P: FnMut(&u8) -> bool>(self, p: P) -> (Span<'a>, Span<'a>) {
        let len = self.bytes().take_while(p).count();
        if len == self.len() {
            (self, Span::default())
        } else {
            self.split_at(len)
        }
    }

    pub fn spans<const SPAN_LENGTH: usize>(&self) -> impl Iterator<Item = Span<'a>> {
        use std::iter;
        let mut cur = self.clone();
        iter::from_fn(move || {
            if cur.len() == 0 {
                None
            } else {
                let (l, r) = cur.split_at(SPAN_LENGTH);
                cur = r;
                Some(l)
            }
        })
    }

    pub fn into_spanned<T>(self, data: T) -> Spanned<'a, T> {
        Spanned::new(self, data)
    }

    pub const fn is_empty(&self) -> bool {
        self.source().is_empty()
    }

    pub fn lines(&self) -> impl Iterator<Item = Span> {
        let first_line = self.line();
        let first_line_colum = self.column();
        self.source()
            .split(|&c| c == b'\n')
            .enumerate()
            .map(move |(ind, l)| {
                Span::from_position(
                    l,
                    (
                        ind as u32 + first_line,
                        if ind == 0 { first_line_colum } else { 0 },
                    ),
                )
            })
    }

    pub fn split_once(&self, pred: impl FnMut(&u8) -> bool) -> Option<(Self, Self)> {
        self.source()
            .iter()
            .position(pred)
            .map(|ind| self.split_at(ind))
    }

    /// merges two spans into one,
    /// * Assumes that other comes after self.
    /// * Assumes that both spans are in the same slice.
    /// * no checks are performed to ensure that the above assumptions are true.
    /// TODO: we can check them by adding an `offset` field that holds the span's offset from the
    /// original slice. we can also make the span more compact by using a 32-bit offset (that can
    /// work with upto 4GB documents) and set the position to be (u16, u16) instead.
    pub fn merge(self, other: Self) -> Self {
        let beg = self.source().as_ptr();
        // split it at the end, we get a slice that is of length 0
        let end = other.source().split_at(other.len()).1;
        let slice = unsafe { slice::from_raw_parts(beg, end.as_ptr() as usize - beg as usize) };
        Self::from_position(slice, (self.line(), self.column()))
    }
}

impl Index<usize> for Span<'_> {
    type Output = u8;
    fn index(&self, index: usize) -> &u8 {
        &self.source()[index]
    }
}

impl Index<Range<usize>> for Span<'_> {
    type Output = [u8];
    fn index(&self, index: Range<usize>) -> &[u8] {
        &self.source()[index]
    }
}

impl Index<RangeFrom<usize>> for Span<'_> {
    type Output = [u8];
    fn index(&self, index: RangeFrom<usize>) -> &[u8] {
        &self.source()[index]
    }
}

impl Index<RangeFull> for Span<'_> {
    type Output = [u8];
    fn index(&self, _index: RangeFull) -> &Self::Output {
        &self.source()[..]
    }
}

impl Index<RangeTo<usize>> for Span<'_> {
    type Output = [u8];
    fn index(&self, index: RangeTo<usize>) -> &[u8] {
        &self.source()[index]
    }
}

impl<'a> AsRef<[u8]> for Span<'a> {
    fn as_ref(&self) -> &[u8] {
        self.source()
    }
}

#[cfg(test)]
mod test {
    use super::Span;

    #[test]
    fn split_at_same_line() {
        let str = br#"this is a test"#;
        let s = Span::new(str);
        assert_eq!(s.source(), str);
        assert_eq!(s.line(), 1);
        assert_eq!(s.column(), 1);

        let (s1, s2) = s.split_at(4);
        assert_eq!(s1.source(), b"this");
        assert_eq!(s1.line(), 1);
        assert_eq!(s1.column(), 1);
        assert_eq!(s2.source(), b" is a test");
        assert_eq!(s2.line(), 1);
        assert_eq!(s2.column(), 5);
    }

    #[test]
    fn split_at_c1_l2() {
        let str = b"this\nis\na\ntest";
        let s = Span::new(str);
        let (s1, s2) = s.split_at(5);
        assert_eq!(s1.source(), b"this\n");
        assert_eq!(s1.line(), 1);
        assert_eq!(s1.column(), 1);
        assert_eq!(s2.source(), b"is\na\ntest");
        assert_eq!(s2.line(), 2);
        assert_eq!(s2.column(), 1);
    }

    #[test]
    fn find() {
        use super::*;
        let text = "this is some text";
        let s = Span::new(text.as_bytes());
        let (s1, s2) = s.find(b"is so").map(|i| (&s[..i], &s[i..])).unwrap();
        assert_eq!(s1, b"this ");
        assert_eq!(s2, b"is some text");
    }

    #[test]
    #[should_panic]
    fn split_at_out_of_bound() {
        let str = b"this is a test";
        Span::new(str).split_at(100);
    }

    #[test]
    fn split_once() {
        let text = b"abcdef ghijk";
        let span = Span::new(text);
        let (s1, s2) = span.split_once(|&c| c == b' ').unwrap();
        assert_eq!(s1.source(), b"abcdef");
        assert_eq!(s1.line(), 1);
        assert_eq!(s1.column(), 1);
        assert_eq!(s2.source(), b" ghijk");
        assert_eq!(s2.line(), 1);
        assert_eq!(s2.column(), 7);
    }

    #[test]
    fn merge() {
        // consecutive slices
        {
            let text = b"abcdefghijklmnopqrstuvwxyz";
            let span = Span::new(text);
            let (s1, s2) = span.split_at(10);
            let s3 = s1.merge(s2);
            assert_eq!(s3.source(), text);
        }

        // empty slice
        {
            let text = b"abcdefghijklmnopqrstuvwxyz";
            let span = Span::new(text);
            let (s1, s2) = span.split_at(0);
            assert_eq!(s1.source(), b"");
            {
                let s3 = s1.merge(s2);
                assert_eq!(s3.source(), text);
            }
            {
                let (s3, s4) = s2.split_at(10);
                {
                    let s5 = s1.merge(s3);
                    assert_eq!(s5.source(), text[..10].as_ref());
                }
                {
                    let s6 = s1.merge(s4);
                    assert_eq!(s6.source(), text);
                }
            }
        }
        {
            let text = b"abcdefghijklmnopqrstuvwxyz";
            let span = Span::new(text);
            let (s1, s2) = span.split_at(span.len());
            assert_eq!(s2.source(), b"");
            assert_eq!(s1.source(), text);
            {
                let s3 = s1.merge(s2);
                assert_eq!(s3.source(), text);
            }
            {
                let (s3, s4) = s1.split_at(10);
                {
                    let s5 = s3.merge(s2);
                    assert_eq!(s5.source(), text);
                }
                {
                    let s6 = s4.merge(s2);
                    assert_eq!(s6.source(), text[10..].as_ref());
                }
            }
        }
    }
}
