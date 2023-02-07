use crate::{log::format_error, span::*};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Error<'a> {
    DecimalLiteral,
    HexLiteral,
    InvalidChar(u8),
    InvalidEscape(u8),
    UnexpectedChar(u8),
    EmptyChar,
    NonAsciiChars,
    BadStringLiteral(Vec<Spanned<'a, Error<'a>>>),
    UnterminatedString,
    UnterminatedComment,
    UnterminatedChar,
}

/// lexer for decaf programming language
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    // keywords
    If,
    Else,
    While,
    For,
    Break,
    Continue,
    Return,
    Int,
    Bool,
    True,
    False,
    Void,
    Len,
    // operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Equal,
    NotEqual,
    And,
    Or,
    Not,
    Question,
    Colon,
    Assign,
    AddAssign,
    SubAssign,
    // delimiters
    Semicolon,
    Comma,
    LeftParen,
    RightParen,
    SquareLeft,
    SquareRight,
    CurlyLeft,
    CurlyRight,
    // literals
    Identifier,
    DecimalLiteral,
    HexLiteral,
    Char(u8),
    String,

    Space,
    LineComment,
    BlockComment,

    // end of file
    Eof,
}

type Result<'a> = std::result::Result<Token, Error<'a>>;

struct StringLiteralBuilder<'a> {
    escaped: bool,
    errors: Vec<Spanned<'a, Error<'a>>>,
}

impl<'a> StringLiteralBuilder<'a> {
    const fn new() -> Self {
        Self {
            escaped: false,
            errors: Vec::new(),
        }
    }

    const fn escape_next(&mut self) {
        self.escaped = true;
    }

    fn build_from_span(span: Span<'a>) -> Spanned<Result> {
        assert!(span.starts_with(b"\""));
        let mut builder = Self::new();
        builder.escape_next();
        let mut builder = span
            .spans::<1>()
            // skip the last char
            .take(span.len() - 1)
            .fold(builder, |mut builder, c| {
                if let Some(err) = builder.update(c[0]) {
                    builder.errors.push(c.into_spanned(err));
                }
                builder
            });

        if !span.ends_with(b"\"") {
            builder
                .errors
                .push(span.into_spanned(Error::UnterminatedString));
        } else if builder.escaped {
            builder
                .errors
                .push(span.into_spanned(Error::UnterminatedString));
        }

        if builder.errors.is_empty() {
            span.into_spanned(Ok(Token::String))
        } else {
            span.into_spanned(Err(Error::BadStringLiteral(builder.errors)))
        }
    }

    fn update(&mut self, c: u8) -> Option<Error<'a>> {
        if self.escaped {
            self.escaped = false;
            if !is_escaped_char(c) {
                Some(Error::InvalidEscape(c))
            } else {
                None
            }
        } else {
            if c == b'\\' {
                self.escaped = true;
                None
            } else if !is_dcf_char(c) {
                Some(Error::InvalidChar(c))
            } else {
                None
            }
        }
    }
}

fn symbol(span: Span) -> Option<(Spanned<Result>, Span)> {
    assert!(span.len() >= 1);
    {
        if span.len() > 1 {
            let (ch, rem) = span.split_at(2);
            match &ch[..] {
                b"<=" => Some((ch.into_spanned(Ok(Token::LessEqual)), rem)),
                b">=" => Some((ch.into_spanned(Ok(Token::GreaterEqual)), rem)),
                b"==" => Some((ch.into_spanned(Ok(Token::Equal)), rem)),
                b"!=" => Some((ch.into_spanned(Ok(Token::NotEqual)), rem)),
                b"+=" => Some((ch.into_spanned(Ok(Token::AddAssign)), rem)),
                b"-=" => Some((ch.into_spanned(Ok(Token::SubAssign)), rem)),
                b"&&" => Some((ch.into_spanned(Ok(Token::And)), rem)),
                b"||" => Some((ch.into_spanned(Ok(Token::Or)), rem)),
                _ => None,
            }
        } else {
            None
        }
    }
    .or_else(|| {
        let (ch, rem) = span.split_at(1);
        match ch[0] {
            b'+' => Some((ch.into_spanned(Ok(Token::Plus)), rem)),
            b'-' => Some((ch.into_spanned(Ok(Token::Minus)), rem)),
            b'*' => Some((ch.into_spanned(Ok(Token::Star)), rem)),
            b'/' => Some((ch.into_spanned(Ok(Token::Slash)), rem)),
            b'%' => Some((ch.into_spanned(Ok(Token::Percent)), rem)),
            b'!' => Some((ch.into_spanned(Ok(Token::Not)), rem)),
            b';' => Some((ch.into_spanned(Ok(Token::Semicolon)), rem)),
            b'<' => Some((ch.into_spanned(Ok(Token::Less)), rem)),
            b'>' => Some((ch.into_spanned(Ok(Token::Greater)), rem)),
            b'=' => Some((ch.into_spanned(Ok(Token::Assign)), rem)),
            b'{' => Some((ch.into_spanned(Ok(Token::CurlyLeft)), rem)),
            b'}' => Some((ch.into_spanned(Ok(Token::CurlyRight)), rem)),
            b'[' => Some((ch.into_spanned(Ok(Token::SquareLeft)), rem)),
            b']' => Some((ch.into_spanned(Ok(Token::SquareRight)), rem)),
            b',' => Some((ch.into_spanned(Ok(Token::Comma)), rem)),
            b'(' => Some((ch.into_spanned(Ok(Token::LeftParen)), rem)),
            b')' => Some((ch.into_spanned(Ok(Token::RightParen)), rem)),
            b'?' => Some((ch.into_spanned(Ok(Token::Question)), rem)),
            b':' => Some((ch.into_spanned(Ok(Token::Colon)), rem)),
            c if !c.is_ascii_alphanumeric() => {
                Some((ch.into_spanned(Err(Error::UnexpectedChar(c))), rem))
            }
            _ => None,
        }
    })
}

fn identifier<'a>(span: Span<'a>) -> Option<(Spanned<Result>, Span<'a>)> {
    assert!(!span.is_empty());
    if !span[0].is_ascii_alphabetic() && span[0] != b'_' {
        None
    } else {
        let keyword = |(span, rem): (Span<'a>, _)| match span.source() {
            b"void" => (span.into_spanned(Ok(Token::Void)), rem),
            b"int" => (span.into_spanned(Ok(Token::Int)), rem),
            b"bool" => (span.into_spanned(Ok(Token::Bool)), rem),
            b"if" => (span.into_spanned(Ok(Token::If)), rem),
            b"for" => (span.into_spanned(Ok(Token::For)), rem),
            b"while" => (span.into_spanned(Ok(Token::While)), rem),
            b"break" => (span.into_spanned(Ok(Token::Break)), rem),
            b"continue" => (span.into_spanned(Ok(Token::Continue)), rem),
            b"return" => (span.into_spanned(Ok(Token::Return)), rem),
            b"len" => (span.into_spanned(Ok(Token::Len)), rem),
            b"true" => (span.into_spanned(Ok(Token::True)), rem),
            b"false" => (span.into_spanned(Ok(Token::False)), rem),
            _ => (span.into_spanned(Ok(Token::Identifier)), rem),
        };
        Some(
            span.split_once(|&c| !c.is_ascii_alphanumeric() && c != b'_')
                .map(keyword)
                .unwrap_or_else(|| keyword((span, span.split_at(span.len()).1))),
        )
    }
}

fn skip_spaces(span: Span) -> Option<(Spanned<Result>, Span)> {
    assert!(!span.is_empty());
    span[0].is_ascii_whitespace().then(|| {
        span.split_once(|&c| !c.is_ascii_whitespace())
            .map(|(spaces, rem)| (spaces.into_spanned(Ok(Token::Space)), rem))
            .unwrap_or((
                span.into_spanned(Ok(Token::Space)),
                span.split_at(span.len()).1,
            ))
    })
}

fn skip_line_comment(span: Span) -> Option<(Spanned<Result>, Span)> {
    assert!(!span.is_empty());
    span.starts_with(b"//").then(|| {
        let (cmt, rem) = span
            .split_once(|&c| c == b'\n')
            .unwrap_or_else(|| (span, span.split_at(span.len()).1));
        (cmt.into_spanned(Ok(Token::LineComment)), rem)
    })
}

fn skip_block_comment(span: Span) -> Option<(Spanned<Result>, Span)> {
    assert!(!span.is_empty());
    if span.starts_with(b"/*") {
        let split = span
            .split_at(2)
            .1
            .find(b"*/")
            .map(|i| span.split_at(i + 4).1);
        if split.is_none() {
            Some((
                span.into_spanned(Err(Error::UnterminatedComment)),
                span.split_at(span.len()).1,
            ))
        } else {
            Some((span.into_spanned(Ok(Token::BlockComment)), split.unwrap()))
        }
    } else {
        None
    }
}

fn int_literal(span: Span) -> Option<(Spanned<Result>, Span)> {
    assert!(!span.is_empty());
    if span[0].is_ascii_digit() {
        let (lit, rem) = span
            .split_once(|c| !c.is_ascii_alphanumeric())
            .unwrap_or(span.split_at(span.len()));
        if lit.starts_with(b"0x") {
            if lit[2..].iter().find(|c| !c.is_ascii_hexdigit()).is_some() {
                Some((lit.into_spanned(Err(Error::HexLiteral)), rem))
            } else {
                Some((lit.into_spanned(Ok(Token::HexLiteral)), rem))
            }
        } else {
            if lit[..].iter().find(|c| !c.is_ascii_digit()).is_some() {
                Some((lit.into_spanned(Err(Error::DecimalLiteral)), rem))
            } else {
                Some((lit.into_spanned(Ok(Token::DecimalLiteral)), rem))
            }
        }
    } else {
        None
    }
}

const fn is_escaped_char(c: u8) -> bool {
    matches!(c, b'n' | b't' | b'\\' | b'\'' | b'"')
}

fn escaped_char<'a>(span: Span<'a>) -> Spanned<Result> {
    assert!(span.len() == 4);
    assert!(span.starts_with(b"'\\"));
    if span[3] != b'\'' {
        span.into_spanned(Err(Error::UnterminatedChar))
    } else {
        let c = span[2];
        match c {
            b'n' => span.into_spanned(Ok(Token::Char(b'\n'))),
            b't' => span.into_spanned(Ok(Token::Char(b'\t'))),
            b'\\' => span.into_spanned(Ok(Token::Char(b'\\'))),
            b'\'' => span.into_spanned(Ok(Token::Char(b'\''))),
            b'"' => span.into_spanned(Ok(Token::Char(b'"'))),
            c => span.into_spanned(Err(Error::InvalidEscape(c))),
        }
    }
}

const fn is_dcf_char(c: u8) -> bool {
    matches!(c, 32..=33 | 35..=38 | 40..=91 | 93..=126)
}

fn dcf_char<'a>(span: Span<'a>) -> Spanned<Result> {
    assert!(span.len() == 3);
    let c = span[1];
    match c {
        c if is_dcf_char(c) => span.into_spanned(Ok(Token::Char(c))),
        _ => span.into_spanned(Err(Error::InvalidChar(c))),
    }
}

fn char_literal(span: Span) -> Option<(Spanned<Result>, Span)> {
    assert!(!span.is_empty());
    if span.len() < 3 || !span.starts_with(b"'") {
        None
    } else if span[1] == b'\\' {
        // escaped char
        if span.len() < 4 {
            Some((
                span.into_spanned(Err(Error::UnterminatedChar)),
                span.split_at(span.len()).1,
            ))
        } else {
            let (lit, rem) = span.split_at(4);
            Some((escaped_char(lit), rem))
        }
    } else {
        if span[1] == b'\'' {
            let (lit, rem) = span.split_at(2);
            Some((lit.into_spanned(Err(Error::EmptyChar)), rem))
        } else if span[2] != b'\'' {
            let (lit, rem) = span.split_at(2);
            Some((lit.into_spanned(Err(Error::UnterminatedChar)), rem))
        } else {
            let (lit, rem) = span.split_at(3);
            Some((dcf_char(lit), rem))
        }
    }
}

fn string_literal(span: Span) -> Option<(Spanned<Result>, Span)> {
    assert!(!span.is_empty());
    if span.starts_with(b"\"") {
        // take the string literal even if it contains errors
        // set `last_char` to '\\' to skip the first quote
        let mut last_char = b'\\';
        let mut break_next = false;
        let (lit, rem) = span.take_while(|c| {
            if break_next {
                false
            } else {
                if *c == b'"' && last_char != b'\\' {
                    break_next = true;
                    true
                } else {
                    last_char = *c;
                    true
                }
            }
        });

        // collect errors in the string literal
        Some((StringLiteralBuilder::build_from_span(lit), rem))
    } else {
        None
    }
}

fn is_ascii(c: &u8) -> bool {
    match c {
        32..=126 | b'\t' | b'\n' | b'\r' => true,
        _ => false
    }
}

/// collect non-ascii chars
fn non_ascii_graphic_chars(span: Span) -> Option<(Spanned<Result>, Span)> {
    assert!(!span.is_empty());
    // this panics if the span is empty
    let (bad_chars, rem) = span.split_once(is_ascii).unwrap();
    if bad_chars.is_empty() {
        None
    } else {
        Some((bad_chars.into_spanned(Err(Error::NonAsciiChars)), rem))
    }
}

fn token(span: Span) -> Option<(Spanned<Result>, Span)> {
    if span.is_empty() {
        return None;
    } else {
        // the non_ascii_graphic_chars has to come before spaces
        // skip_spaces skips some of illegal chars
        non_ascii_graphic_chars(span)
            .or_else(|| skip_spaces(span))
            .or_else(|| skip_line_comment(span))
            .or_else(|| skip_block_comment(span))
            .or_else(|| identifier(span))
            .or_else(|| int_literal(span))
            .or_else(|| char_literal(span))
            .or_else(|| string_literal(span))
            .or_else(|| symbol(span))
    }
}

pub fn tokens<L: FnMut(Spanned<Error>)>(
    text: &[u8],
    mut log: L,
) -> impl Iterator<Item = Spanned<Result>> {
    use std::iter;
    let mut s = Span::new(text);
    iter::from_fn(move || {
        if s.is_empty() {
            return None;
        } else {
            let (tok, rem) = token(s)?;
            s = rem;
            Some(tok)
        }
    })
    .filter(|t| {
        !matches!(
            t.get(),
            Ok(Token::Space) | Ok(Token::LineComment) | Ok(Token::BlockComment)
        )
    })
    .inspect(move |tok| {
        if let Err(err) = tok.get() {
            log(tok.span().into_spanned(err.clone()))
        }
    })
    .chain(iter::once(s.into_spanned(Ok(Token::Eof))))
}

/// creats a log function to the given stream.
/// example:
/// TODO: fix this
/// ```
/// // use dcfrs::lexer::{log_err, tokens};
///
/// // let mut mock_stderr = vec![];
/// // let mut log = log_err(&mut mock_stderr, "test.dcf");
///
/// // tokens(b"'\\a'", log);
/// // assert_eq!(mock_stderr, b"test.dcf:1:2: invalid escape sequence: \\a");
/// ```
pub fn log_err<'a, T: AsRef<str> + 'a>(
    mut write: impl FnMut(String) + 'a,
    input_file: T,
) -> impl FnMut(Spanned<Error>) + 'a {
    move |err| {
        let string = |slice: &[u8]| String::from_utf8(slice.to_vec()).unwrap();
        let mut loge =
            |pos: (u32, u32), msg: &str| write(format_error(input_file.as_ref(), pos, msg));

        fn print_u8(c: u8) -> String {
            if c.is_ascii_digit() {
                format!("{}", c as char)
            } else {
                format!("\\x{:02x}", c)
            }
        }
        let mut handle_single_error = |err: Spanned<Error>| match err.get() {
            Error::HexLiteral => {
                loge(
                    err.position(),
                    &format!("invalid hex literal: {}", string(err.fragment())),
                );
            }
            Error::DecimalLiteral => {
                loge(
                    err.position(),
                    &format!("invalid decimal literal: {}", string(err.fragment())),
                );
            }
            Error::EmptyChar => {
                loge(err.position(), "empty char literal");
            }
            Error::InvalidEscape(c) => {
                loge(
                    err.position(),
                    &format!("invalid escape sequence: \\{}", print_u8(*c)),
                );
            }
            Error::InvalidChar(c) => {
                loge(
                    err.position(),
                    &format!("invalid character literal: {}", print_u8(*c)),
                );
            }
            Error::UnexpectedChar(c) => {
                loge(
                    err.position(),
                    &format!("unexpected character: {}", print_u8(*c)),
                );
            }
            Error::UnterminatedString => {
                loge(err.position(), "unterminated string literal");
            }
            Error::UnterminatedChar => {
                loge(err.position(), "unterminated char literal");
            }
            Error::UnterminatedComment => {
                loge(err.position(), "unterminated block comment");
            }
            Error::NonAsciiChars => {
                loge(
                    err.position(),
                    &format!("non-ascii characters: {}", string(err.fragment())),
                );
            }
            _ => unreachable!(),
        };

        match err.get() {
            Error::BadStringLiteral(errs) => {
                errs.into_iter()
                    .for_each(|err| handle_single_error(err.clone()));
            }
            _ => handle_single_error(err),
        };
    }
}

#[cfg(test)]
mod test {
    use super::Token::*;
    use super::*;

    // fn parsed<'a>(opt: Option<(Spanned<'a, Result<'a>>, Span<'a>)>) -> Spanned<'a, Result<'a>> {
    //     opt.unwrap().0
    // }

    fn rem<'a>(opt: Option<(Spanned<'a, Result<'a>>, Span<'a>)>) -> Span<'a> {
        opt.unwrap().1
    }

    #[test]
    fn identifier() {
        use super::*;
        let text = b"abc";
        let span = Span::new(text);
        let (s1, s2) = identifier(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), Identifier);
        assert_eq!(s1.fragment(), b"abc");
        assert_eq!(s2.source(), b"");

        let text = b"_abc";
        let span = Span::new(text);
        let (s1, s2) = identifier(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), Identifier);
        assert_eq!(s1.fragment(), b"_abc");
        assert_eq!(s2.source(), b"");

        let text = b"abc def";
        let span = Span::new(text);
        let (s1, s2) = identifier(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), Identifier);
        assert_eq!(s1.fragment(), b"abc");
        assert_eq!(s2.source(), b" def");

        let text = b"123abc";
        assert!(identifier(Span::new(text)).is_none());
    }

    #[test]
    fn char_literal() {
        use super::*;
        let text = b"'a'";
        let span = Span::new(text);
        let (s1, s2) = char_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), Char(b'a'));
        assert_eq!(s1.fragment(), b"'a'");
        assert_eq!(s2.source(), b"");

        let text = b"'\\'";
        let span = Span::new(text);
        let (s1, s2) = char_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap_err(), Error::UnterminatedChar);
        assert_eq!(s1.fragment(), b"'\\'");
        assert_eq!(s2.source(), b"");

        let text = b"'	'";
        let span = Span::new(text);
        let (s1, s2) = char_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap_err(), Error::InvalidChar(b'\t'));
        assert_eq!(s1.fragment(), b"'	'");
        assert_eq!(s2.source(), b"");

        let text = b"'\\t'";
        let span = Span::new(text);
        let (s1, s2) = char_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), Char(b'\t'));
        assert_eq!(s1.fragment(), b"'\\t'");
        assert_eq!(s2.source(), b"");
    }

    #[test]
    fn string_literal() {
        use super::*;
        let text = b"\"abc\"";
        let span = Span::new(text);
        let (s1, s2) = string_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), String);
        assert_eq!(s1.fragment(), b"\"abc\"");
        assert_eq!(s2.source(), b"");

        let text = br#""\"abcdef\"""#;
        let span = Span::new(text);
        let (s1, s2) = string_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), String,);
        assert_eq!(s1.fragment(), br#""\"abcdef\"""#);
        assert_eq!(s2.source(), b"");

        let text = b"\"abc alot of text that does not\\\" terminate with a quote";
        let span = Span::new(text);
        let (s1, s2) = string_literal(span).unwrap();
        s1.get().clone().unwrap_err();
        assert_eq!(
            s1.fragment(),
            b"\"abc alot of text that does not\\\" terminate with a quote"
        );
        assert_eq!(s2.source(), b"");
    }

    #[test]
    fn int_literal() {
        use super::*;
        let text = b"123";
        let span = Span::new(text);
        let (s1, s2) = int_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), DecimalLiteral);
        assert_eq!(s1.fragment(), b"123");
        assert_eq!(s2.source(), b"");

        let text = b"123abc";
        let span = Span::new(text);
        let (s1, s2) = int_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap_err(), Error::DecimalLiteral);
        assert_eq!(s1.fragment(), b"123abc");
        assert_eq!(s2.source(), b"");

        let text = b"12a111";
        let span = Span::new(text);
        let (s1, s2) = int_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap_err(), Error::DecimalLiteral);
        assert_eq!(s1.fragment(), b"12a111");
        assert_eq!(s2.source(), b"");

        let text = b"b1111";
        assert!(int_literal(Span::new(text)).is_none());

        let text = b"0x123";
        let span = Span::new(text);
        let (s1, s2) = int_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), HexLiteral);
        assert_eq!(s1.fragment(), b"0x123");
        assert_eq!(s2.source(), b"");

        let text = b"0x123abc";
        let span = Span::new(text);
        let (s1, s2) = int_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), HexLiteral);
        assert_eq!(s1.fragment(), b"0x123abc");
        assert_eq!(s2.source(), b"");

        let text = b"0x123abcg";
        let span = Span::new(text);
        let (s1, s2) = int_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap_err(), Error::HexLiteral);
        assert_eq!(s1.fragment(), b"0x123abcg");
        assert_eq!(s2.source(), b"");

        let text = b"0x12gabcg";
        let span = Span::new(text);
        let (s1, s2) = int_literal(span).unwrap();
        assert_eq!(s1.get().clone().unwrap_err(), Error::HexLiteral);
        assert_eq!(s1.fragment(), b"0x12gabcg");
        assert_eq!(s2.source(), b"");
    }

    #[test]
    fn skip_spaces() {
        use super::*;
        let span = Span::new(b"    some text");
        assert_eq!(rem(skip_spaces(span)).source(), b"some text");
        let span = Span::new(b"\n\n\t  \n some text");
        assert_eq!(rem(skip_spaces(span)).source(), b"some text");
        let span = Span::new(b"   \n\t");
        assert_eq!(rem(skip_spaces(span)).source(), b"");
    }

    #[test]
    fn skip_line_comment() {
        use super::*;
        let span = Span::new(b"// comment\nsometext");
        let span = skip_line_comment(span);
        assert_eq!(rem(span).source(), b"\nsometext");
        let span = Span::new(b"// comment");
        let span = skip_line_comment(span);
        assert_eq!(rem(span).source(), b"");
    }

    #[test]
    fn skip_block_comment() {
        use super::*;
        let span = Span::new(b"/* comment */sometext");
        let span = skip_block_comment(span);
        assert_eq!(rem(span).source(), b"sometext",);
        // TODO: check how to see the logged output
        let span = Span::new(b"/* comment ");
        let span = skip_block_comment(span);
        assert_eq!(rem(span).source(), b"");

        let span = Span::new(b"/**/");
        let span = skip_block_comment(span);
        assert_eq!(rem(span).source(), b"");

        let span = Span::new(b"/*/");
        let rem = rem(skip_block_comment(span));
        assert!(rem.is_empty())
    }

    #[test]
    fn symbol() {
        use super::*;
        let text = b"==";
        let span = Span::new(text);
        let (s1, s2) = symbol(span).unwrap();
        assert_eq!(s1.get().clone().unwrap(), Equal);
        assert_eq!(s1.fragment(), b"==");
        assert_eq!(s2.source(), b"");

        use std::iter;
        let text = b"=+-*/%&&||!<>?:";
        let mut span = Span::new(text);
        let symbols = iter::from_fn(move || {
            if span.len() == 0 {
                None
            } else {
                let (l, r) = symbol(span).unwrap();
                span = r;
                Some(l.get().clone().unwrap())
            }
        })
        .collect::<Vec<_>>();
        assert_eq!(
            symbols,
            vec![
                Assign, Plus, Minus, Star, Slash, Percent, And, Or, Not, Less, Greater, Question,
                Colon,
            ]
        )
    }

    // #[test]
    // fn non_ascii_graphic_chars() {
    //     use super::*;
    //     let text = "αβγδεζηθικλμνξοπρστυφχψω".as_bytes();
    //     let span = Span::new(text);
    //     let (s1, s2) = non_ascii_graphic_chars(span).unwrap();
    //     assert_eq!(s1.get().clone().unwrap(), Error::NonAsciiChars);
    // }
}
