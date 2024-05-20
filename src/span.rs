use std::{
    fmt::Debug,
    hash::Hash,
    ops::{Index, Range, RangeFrom, RangeFull, RangeTo},
    slice,
};

#[derive(Clone, PartialEq, Eq)]
pub struct SpanSource<'a> {
    source: &'a str,
    lines: Vec<*const u8>,
    lengths: Vec<usize>,
}

impl<'a> SpanSource<'a> {
    pub fn new(source: &'a str) -> Self {
        let lines = source
            .split(|c| c == '\n')
            .map(|line| line.as_ptr())
            .collect();
        let lengths = source.split(|c| c == '\n').map(|line| line.len()).collect();
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

    pub fn line(&self, line_num: usize) -> Option<&str> {
        assert!(line_num > 0);
        self.lines.get(line_num - 1).map(|&line| unsafe {
            core::str::from_utf8_unchecked(slice::from_raw_parts(line, self.lengths[line_num - 1]))
        })
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

    pub const fn fragment(&self) -> &'a str {
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
    source: &'a str,
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
        self.source().to_string()
    }
}

impl<'a> Span<'a> {
    pub fn position(&self) -> (usize, usize) {
        (self.line(), self.column())
    }

    pub fn first(&self) -> Option<char> {
        self.source.chars().next()
    }

    pub fn chars(&self) -> impl Iterator<Item = char> + '_ {
        self.source().chars()
    }

    pub fn as_str(&self) -> &'a str {
        self.source()
    }

    pub const fn source(&self) -> &'a str {
        self.source
    }

    pub fn bytes(&self) -> impl Iterator<Item = u8> + 'a {
        self.source().bytes()
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

    pub fn find(&self, pat: &str) -> Option<usize> {
        self.source.find(pat)
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

    pub fn starts_with(&self, pat: &str) -> bool {
        self.source().starts_with(pat)
    }

    pub fn ends_with(&self, pat: &str) -> bool {
        self.source().ends_with(pat)
    }

    pub fn take_while<P: FnMut(char) -> bool>(self, mut p: P) -> (Span<'a>, Span<'a>) {
        self.split_once(|c| !p(c)).unwrap_or((
            self,
            Span {
                source: self.source().split_at(0).0,
                span_source: self.span_source,
            },
        ))
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

    pub fn split_once(&self, pred: impl FnMut(char) -> bool) -> Option<(Self, Self)> {
        if let Some(offset) = self.source().find(pred) {
            Some(self.split_at(offset))
        } else {
            None
        }
    }

    pub fn offset(&self) -> usize {
        unsafe {
            self.source()
                .as_ptr()
                .offset_from(self.span_source.source.as_ptr()) as usize
        }
    }

    pub fn merge(self, other: Self) -> Self {
        assert!(self.span_source == other.span_source);
        let beg = self.offset();
        let end = other.offset() + other.len();
        Self {
            source: &self.span_source.source[beg..end],
            span_source: self.span_source,
        }
    }
}

impl Index<Range<usize>> for Span<'_> {
    type Output = str;
    fn index(&self, index: Range<usize>) -> &str {
        &self.source()[index]
    }
}

impl Index<RangeFrom<usize>> for Span<'_> {
    type Output = str;
    fn index(&self, index: RangeFrom<usize>) -> &str {
        &self.source()[index]
    }
}

impl Index<RangeFull> for Span<'_> {
    type Output = str;
    fn index(&self, _index: RangeFull) -> &Self::Output {
        self.source()
    }
}

impl Index<RangeTo<usize>> for Span<'_> {
    type Output = str;
    fn index(&self, index: RangeTo<usize>) -> &str {
        &self.source()[index]
    }
}

impl<'a> AsRef<str> for Span<'a> {
    fn as_ref(&self) -> &str {
        self.source()
    }
}

#[cfg(test)]
mod test {
    use crate::span::SpanSource;

    #[test]
    fn split_at_same_line() {
        let str = r#"this is a test"#;
        let span_source = SpanSource::new(str);
        let s = span_source.source();
        assert_eq!(s.source(), str);
        assert_eq!(s.line(), 1);
        assert_eq!(s.column(), 1);

        let (s1, s2) = s.split_at(4);
        assert_eq!(s1.source(), "this");
        assert_eq!(s1.line(), 1);
        assert_eq!(s1.column(), 1);
        assert_eq!(s2.source(), " is a test");
        assert_eq!(s2.line(), 1);
        assert_eq!(s2.column(), 5);
    }

    #[test]
    fn split_at_c1_l2() {
        let str = "this\nis\na\ntest";
        let span_source = SpanSource::new(str);
        let s = span_source.source();
        let (s1, s2) = s.split_at(5);
        assert_eq!(s1.source(), "this\n");
        assert_eq!(s1.line(), 1);
        assert_eq!(s1.column(), 1);
        assert_eq!(s2.source(), "is\na\ntest");
        assert_eq!(s2.line(), 2);
        assert_eq!(s2.column(), 1);
    }

    #[test]
    fn find() {
        use super::*;
        let text = "this is some text";
        let span_source = SpanSource::new(text);
        let s = span_source.source();
        let (s1, s2) = s.find("is so").map(|i| (&s[..i], &s[i..])).unwrap();
        assert_eq!(s1, "this ");
        assert_eq!(s2, "is some text");
    }

    #[test]
    #[should_panic]
    fn split_at_out_of_bound() {
        let str = "this is a test";
        let span_source = SpanSource::new(str);
        span_source.source().split_at(100);
    }

    #[test]
    fn split_once() {
        let text = "abcdef ghijk";
        let span_source = SpanSource::new(text);
        let span = span_source.source();
        let (s1, s2) = span.split_once(|c| c == ' ').unwrap();
        assert_eq!(s1.source(), "abcdef");
        assert_eq!(s1.line(), 1);
        assert_eq!(s1.column(), 1);
        assert_eq!(s2.source(), " ghijk");
        assert_eq!(s2.line(), 1);
        assert_eq!(s2.column(), 7);
    }

    #[test]
    fn merge() {
        // consecutive slices
        {
            let text = "abcdefghijklmnopqrstuvwxyz";
            let span_source = SpanSource::new(text);
            let span = span_source.source();
            let (s1, s2) = span.split_at(10);
            let s3 = s1.merge(s2);
            assert_eq!(s3.source(), text);
        }

        // empty slice
        {
            let text = "abcdefghijklmnopqrstuvwxyz";
            let span_source = SpanSource::new(text);
            let span = span_source.source();
            let (s1, s2) = span.split_at(0);
            assert_eq!(s1.source(), "");
            {
                let s3 = s1.merge(s2);
                assert_eq!(s3.source(), text);
            }
            {
                let (s3, s4) = s2.split_at(10);
                {
                    let s5 = s1.merge(s3);
                    assert_eq!(s5.source(), &text[..10]);
                }
                {
                    let s6 = s1.merge(s4);
                    assert_eq!(s6.source(), text);
                }
            }
        }
        {
            let text = "abcdefghijklmnopqrstuvwxyz";
            let span_source = SpanSource::new(text);
            let span = span_source.source();
            let (s1, s2) = span.split_at(span.len());
            assert_eq!(s2.source(), "");
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
                    assert_eq!(s6.source(), &text[10..]);
                }
            }
        }
    }
}
