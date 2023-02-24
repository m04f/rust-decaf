use std::{
    fmt::Debug,
    hash::Hash,
    iter::Copied,
    ops::{Index, Range, RangeFrom, RangeFull, RangeTo},
    slice,
};

#[derive(Clone, PartialEq, Eq)]
pub struct SpanSource<'a> {
    source: &'a [u8],
    lines: Vec<*const u8>,
    lengths: Vec<usize>,
}

impl<'a> SpanSource<'a> {
    pub fn new(source: &'a [u8]) -> Self {
        let lines = source
            .split(|c| *c == b'\n')
            .map(|line| line.as_ptr())
            .collect();
        let lengths = source
            .split(|c| *c == b'\n')
            .map(|line| line.len())
            .collect();
        Self {
            source,
            lines,
            lengths,
        }
    }

    pub fn get_line(&self, span: Span) -> Span {
        assert!(span.span_source == self);
        Span {
            source: self
                .line(
                    self.lines
                        .binary_search(&span.source.as_ptr())
                        .unwrap_or_else(|i| i),
                )
                .unwrap(),
            span_source: self,
        }
    }

    pub fn line(&self, line_num: usize) -> Option<&[u8]> {
        assert!(line_num > 0);
        self.lines
            .get(line_num - 1)
            .map(|&line| unsafe { slice::from_raw_parts(line, self.lengths[line_num - 1]) })
    }

    pub fn get_line_number(&self, span: Span<'a>) -> usize {
        self.lines
            .binary_search(&span.source.as_ptr())
            .map(|i| i + 1)
            .unwrap_or_else(|i| i)
    }

    pub fn get_column(&self, span: Span<'a>) -> usize {
        let line_num = self.get_line_number(span);
        let line = self.line(line_num).unwrap();
        span.source().as_ptr() as usize - line.as_ptr() as usize + 1
    }

    pub fn get_pos(&self, span: Span<'a>) -> (usize, usize) {
        (self.get_line_number(span), self.get_column(span))
    }

    pub fn lines(&self) -> impl Iterator<Item = &[u8]> {
        self.lines
            .iter()
            .zip(self.lengths.iter())
            .map(|(address, len)| unsafe { slice::from_raw_parts(*address, *len) })
    }

    pub fn source(&self) -> Span {
        Span {
            span_source: self,
            source: self.source,
        }
    }
}

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

    pub fn line(&self) -> usize {
        self.span().line()
    }

    pub fn column(&self) -> usize {
        self.span().column()
    }

    pub fn position(&self) -> (usize, usize) {
        self.span().position()
    }

    pub fn into_parts(self) -> (T, Span<'a>) {
        (self.data, self.span)
    }
}

impl<'a, T, E> Spanned<'a, Result<T, E>> {
    // converts Spanned<Result<T, E>> to Result<Spanned<T>, Spanned<E>>
    pub fn transpose(self) -> Result<Spanned<'a, T>, Spanned<'a, E>> {
        self.data
            .map(|data| Spanned::new(self.span, data))
            .map_err(|err| Spanned::new(self.span, err))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Span<'a> {
    source: &'a [u8],
    span_source: &'a SpanSource<'a>,
}

impl Hash for Span<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.source.hash(state);
    }
}

impl<'a> Debug for Span<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl ToString for Span<'_> {
    fn to_string(&self) -> String {
        std::str::from_utf8(self.source()).unwrap().to_string()
    }
}

impl<'a> Span<'a> {
    pub fn position(&self) -> (usize, usize) {
        (self.line(), self.column())
    }

    pub const fn source(&self) -> &'a [u8] {
        self.source
    }

    pub fn bytes(&self) -> Copied<slice::Iter<u8>> {
        self.source().iter().copied()
    }

    pub fn line(&self) -> usize {
        self.span_source.get_line_number(*self)
    }

    pub fn column(&self) -> usize {
        self.span_source.get_column(*self)
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
        (
            Self {
                source: left,
                span_source: self.span_source,
            },
            Self {
                source: right,
                span_source: self.span_source,
            },
        )
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
            self.split_at(self.len())
        } else {
            self.split_at(len)
        }
    }

    pub fn spans<const SPAN_LENGTH: usize>(&self) -> impl Iterator<Item = Span<'a>> {
        use std::iter;
        let mut cur = *self;
        iter::from_fn(move || {
            if cur.is_empty() {
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
    pub fn merge(self, other: Self) -> Self {
        assert!(self.span_source == other.span_source);
        let beg = self.source().as_ptr();
        // split it at the end, we get a slice that is of length 0
        let end = other.source().split_at(other.len()).1;
        let slice = unsafe { slice::from_raw_parts(beg, end.as_ptr() as usize - beg as usize) };
        Self {
            source: slice,
            span_source: self.span_source,
        }
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
        self.source()
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
    use crate::span::SpanSource;

    #[test]
    fn split_at_same_line() {
        let str = br#"this is a test"#;
        let span_source = SpanSource::new(str);
        let s = span_source.source();
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
        let span_source = SpanSource::new(str);
        let s = span_source.source();
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
        let span_source = SpanSource::new(text.as_bytes());
        let s = span_source.source();
        let (s1, s2) = s.find(b"is so").map(|i| (&s[..i], &s[i..])).unwrap();
        assert_eq!(s1, b"this ");
        assert_eq!(s2, b"is some text");
    }

    #[test]
    #[should_panic]
    fn split_at_out_of_bound() {
        let str = b"this is a test";
        let span_source = SpanSource::new(str);
        span_source.source().split_at(100);
    }

    #[test]
    fn split_once() {
        let text = b"abcdef ghijk";
        let span_source = SpanSource::new(text);
        let span = span_source.source();
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
            let span_source = SpanSource::new(text);
            let span = span_source.source();
            let (s1, s2) = span.split_at(10);
            let s3 = s1.merge(s2);
            assert_eq!(s3.source(), text);
        }

        // empty slice
        {
            let text = b"abcdefghijklmnopqrstuvwxyz";
            let span_source = SpanSource::new(text);
            let span = span_source.source();
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
            let span_source = SpanSource::new(text);
            let span = span_source.source();
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
