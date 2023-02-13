use crate::ast::{self, BlockBuilder, Op, Type};

use crate::span::*;

use crate::lexer::Token;

#[cfg(test)]
mod test;

mod result;

use self::result::Error;
use self::result::Error::*;
use self::result::Result;
use self::result::Result::*;

use core::iter::Peekable;
use std::fmt::Debug;

#[derive(Debug, PartialEq, Eq)]
enum Or<T1, T2> {
    First(T1),
    Second(T2),
}

pub struct Parser<'a, I: Iterator<Item = Spanned<'a, Token>>, EH: FnMut(Spanned<'a, Error>)> {
    tokens: Peekable<I>,
    error_callback: EH,
}

macro_rules! binop {
    ($name:ident, $sub:ident, $first:ident => $first_mapped:ident, $($token:ident => $token_mapped:ident),*) => {
        fn $name(&mut self) -> Result<Expr<'a>> {
            let op = move |p: &mut Parser<'a, I, EH>| {
                Self::exact_token(Token::$first)(p)
                    .map(|_| Op::$first_mapped)
                    $(.or_else(|| Self::exact_token(Token::$token)(p).map(|_| Op::$token_mapped)))*
            };
            Self::fold(
                move |p| p.span(move |p| p.cascade(op, Self::$sub, |p| {
                    p.cur_span().into_spanned(ExpectedExpression)
                })),
                Self::$sub,
                |lhs, rhs| { let ((op, rhs), end) = rhs.into_parts(); let beg = *lhs.span(); Expr::binop(lhs, op, rhs, beg.merge(end))},
            )(self)
        }
    };
}

type Expr<'a> = ast::Expr<Span<'a>>;
type Stmt<'a> = ast::Stmt<Span<'a>>;
type Block<'a> = ast::Block<Span<'a>>;
type BlockElem<'a> = ast::BlockElem<Span<'a>>;
type Identifier<'a> = ast::Identifier<Span<'a>>;
type Import<'a> = ast::Import<Span<'a>>;
type IntLiteral<'a> = ast::IntLiteral<Span<'a>>;
type ELiteral<'a> = ast::ELiteral<Span<'a>>;
type BoolLiteral<'a> = ast::BoolLiteral<Span<'a>>;
type CharLiteral<'a> = ast::CharLiteral<Span<'a>>;
type StringLiteral<'a> = ast::StringLiteral<Span<'a>>;
type Arg<'a> = ast::Arg<Span<'a>>;
type Function<'a> = ast::Function<Span<'a>>;
type Var<'a> = ast::Var<Span<'a>>;
type Call<'a> = ast::Call<Span<'a>>;
type Loc<'a> = ast::Loc<Span<'a>>;
type Assign<'a> = ast::Assign<Span<'a>>;
type AssignExpr<'a> = ast::AssignExpr<Span<'a>>;
type DocElem<'a> = ast::DocElem<Span<'a>>;

impl<'a, I: Iterator<Item = Spanned<'a, Token>>, EH: FnMut(Spanned<'a, Error>)> Parser<'a, I, EH> {
    pub fn new(tokens: I, eh: EH) -> Self {
        Self {
            tokens: tokens.peekable(),
            error_callback: eh,
        }
    }

    fn peek(&mut self) -> Token {
        self.tokens.peek().map(|t| *t.get()).unwrap()
    }

    pub fn finised(&mut self) -> bool {
        self.peek() == Token::Eof
    }

    /// returns the next token and advances the iterator.
    /// panics if the iterator is empty.
    fn bump(&mut self) -> Spanned<'a, Token> {
        self.tokens.next().unwrap()
    }

    fn exact_token(
        token: Token,
    ) -> impl FnMut(&mut Parser<'a, I, EH>) -> Result<Spanned<'a, Token>> {
        move |p: &mut Self| match p.peek() {
            Token::CharLiteral(_) => {
                if let Token::CharLiteral(_) = token {
                    Parsed(p.bump())
                } else {
                    Nil
                }
            }
            t if t == token => Parsed(p.bump()),
            _ => Nil,
        }
    }

    fn delimited<O>(
        &mut self,
        left: Token,
        parser: impl FnOnce(&mut Parser<'a, I, EH>) -> Result<O>,
        right: Token,
        get_error: impl FnOnce(&mut Parser<'a, I, EH>) -> Spanned<'a, Error>,
    ) -> Result<O> {
        let mut left_parser = Self::exact_token(left);
        let mut right_parser = Self::exact_token(right);
        let beg = self.cur_span();
        if left_parser(self).is_nil() {
            Nil
        } else {
            let result = parser(self);
            if result.has_output() {
                if right_parser(self).is_nil() {
                    // FIXME: the span is larger since it looks ahead instead of returning the
                    // last position.
                    let end = self.cur_span();
                    (self.error_callback)(
                        beg.merge(end).into_spanned(ExpectedMatching(left, right)),
                    );
                }
            }
            if result.is_nil() {
                let err = get_error(self);
                (self.error_callback)(err);
                right_parser(self);
                Error
            } else {
                result
            }
        }
    }

    fn opt<O>(
        &mut self,
        parser: impl FnOnce(&mut Parser<'a, I, EH>) -> Result<O>,
    ) -> Result<Option<O>> {
        match parser(self) {
            Parsed(o) => Parsed(Some(o)),
            ErrorWithResult(o) => ErrorWithResult(Some(o)),
            Error => Error,
            Nil => Parsed(None),
        }
    }

    fn span<O>(
        &mut self,
        parser: impl FnOnce(&mut Parser<'a, I, EH>) -> Result<O>,
    ) -> Result<Spanned<'a, O>> {
        let beg = self.cur_span();
        match parser(self) {
            Parsed(o) => Parsed(beg.merge(self.cur_span()).into_spanned(o)),
            ErrorWithResult(o) => ErrorWithResult(beg.merge(self.cur_span()).into_spanned(o)),
            Error => Error,
            Nil => Nil,
        }
    }

    fn cascade<O1, O2>(
        &mut self,
        left: impl FnOnce(&mut Parser<'a, I, EH>) -> Result<O1>,
        right: impl FnOnce(&mut Parser<'a, I, EH>) -> Result<O2>,
        right_error: impl FnOnce(&mut Parser<'a, I, EH>) -> Spanned<'a, Error>,
    ) -> Result<(O1, O2)> {
        let left_result = left(self);
        match left_result {
            Nil => Nil,
            Parsed(o1) => match right(self) {
                Nil => {
                    let error = right_error(self);
                    (self.error_callback)(error);
                    Error
                }
                Parsed(o2) => Parsed((o1, o2)),
                ErrorWithResult(o2) => ErrorWithResult((o1, o2)),
                Error => Error,
            },
            Error => Error,
            ErrorWithResult(o1) => match right(self) {
                Nil => {
                    let error = right_error(self);
                    (self.error_callback)(error);
                    Error
                }
                Parsed(o2) => ErrorWithResult((o1, o2)),
                ErrorWithResult(o2) => ErrorWithResult((o1, o2)),
                Error => Error,
            },
        }
    }

    fn preceded<O>(
        &mut self,
        token: Token,
        parser: impl FnOnce(&mut Parser<'a, I, EH>) -> Result<O>,
        get_error: impl FnOnce(&mut Parser<'a, I, EH>) -> Spanned<'a, Error>,
    ) -> Result<O> {
        if Self::exact_token(token)(self).is_nil() {
            Nil
        } else {
            let res = parser(self);
            if res.is_nil() {
                let error = get_error(self);
                (self.error_callback)(error);
                Error
            } else {
                res
            }
        }
    }

    fn terminated<O>(
        &mut self,
        parser: impl FnOnce(&mut Parser<'a, I, EH>) -> Result<O>,
        token: Token,
    ) -> Result<O> {
        let mut token_parser = Self::exact_token(token);
        let parsed = parser(self);
        if parsed.has_output() {
            if token_parser(self).is_nil() {
                let error = self.cur_span().into_spanned(Expected(self.peek(), token));
                (self.error_callback)(error);
                ErrorWithResult(parsed.unwrap_output())
            } else {
                parsed
            }
        } else {
            parsed
        }
    }

    fn ident(&mut self) -> Result<Identifier<'a>> {
        Self::exact_token(Token::Identifier)(self).map(|t| Identifier::from_span(t.span()))
    }

    fn cur_span(&mut self) -> Span<'a> {
        self.tokens.peek().unwrap().span()
    }

    /// does not consume the trailing semicolon.
    fn import(&mut self) -> Result<Import<'a>> {
        self.span(|p| {
            p.delimited(Token::Import, Self::ident, Token::Semicolon, |p| {
                p.cur_span()
                    .into_spanned(Expected(p.peek(), Token::Identifier))
            })
        })
        .map(|spanned_id| Import::from_ident(spanned_id))
    }

    fn len_expr(&mut self) -> Result<Expr<'a>> {
        self.span(|p| {
            p.preceded(
                Token::Len,
                |p| {
                    p.delimited(Token::LeftParen, Self::ident, Token::RightParen, |p| {
                        p.cur_span()
                            .into_spanned(Expected(p.peek(), Token::Identifier))
                    })
                },
                |o1| o1.cur_span().into_spanned(LenNoArg),
            )
        })
        .map(|spanned_len| {
            let (name, span) = spanned_len.into_parts();
            Expr::len(name, span)
        })
    }

    fn neg(&mut self) -> Result<Expr<'a>> {
        self.span(|p| {
            p.preceded(Token::Minus, Self::unit_expr, |p| {
                p.cur_span().into_spanned(ExpectedExpression)
            })
        })
        .map(|s| s.into_parts())
        .map(|(expr, span)| Expr::neg(expr, span))
    }

    fn not(&mut self) -> Result<Expr<'a>> {
        self.span(|p| {
            p.preceded(Token::Not, Self::unit_expr, |p| {
                p.cur_span().into_spanned(ExpectedExpression)
            })
        })
        .map(|s| s.into_parts())
        .map(|(expr, span)| Expr::not(expr, span))
    }

    fn opt_index(&mut self) -> Result<Option<Expr<'a>>> {
        self.opt(|p| {
            p.delimited(Token::SquareLeft, Self::expr, Token::SquareRight, |p| {
                p.cur_span().into_spanned(ExpectedExpression)
            })
        })
    }

    fn call_or_loc(&mut self) -> Result<Or<Call<'a>, Loc<'a>>> {
        let beg = self.cur_span();
        let ident = self.ident();
        if ident.has_output() {
            match self.peek() {
                Token::LeftParen => {
                    let args = self.call_args();
                    ident.and(args).map(|(name, args)| {
                        Or::First(Call::new(name, args, beg.merge(self.cur_span())))
                    })
                }
                Token::SquareLeft => {
                    let index = self.opt_index();
                    ident.and(index).map(|(ident, index)| {
                        Or::Second(Loc::with_offset(
                            ident,
                            index.unwrap(),
                            beg.merge(self.cur_span()),
                        ))
                    })
                }
                _ => ident.map(|ident| Or::Second(Loc::from_ident(ident))),
            }
        } else {
            ident.map(|ident| Or::Second(Loc::from_ident(ident)))
        }
    }

    fn opt_size(&mut self) -> Result<Option<IntLiteral<'a>>> {
        self.opt(|p| {
            p.delimited(
                Token::SquareLeft,
                Self::int_literal,
                Token::SquareRight,
                |p| {
                    p.cur_span()
                        .into_spanned(Expected(p.peek(), Token::DecimalLiteral))
                },
            )
        })
    }

    fn var_type(&mut self) -> Result<Type> {
        Self::exact_token(Token::Int)(self)
            .map(|_| Type::int_type())
            .or_else(|| Self::exact_token(Token::Int)(self).map(|_| Type::bool_type()))
    }

    fn expect(&mut self, token: Token) -> Spanned<'a, Error> {
        self.cur_span().into_spanned(Expected(self.peek(), token))
    }

    fn func_param(&mut self) -> Result<Var<'a>> {
        self.var_type().and_then(|ty| {
            self.ident()
                .map_nil(|| {
                    let err = self.expect(Token::Identifier);
                    (self.error_callback)(err);
                    Error
                })
                .map(|ident| Var::scalar(ty, ident))
        })
        // Self::cascade(Self::var_type, Self::ident, |p| {
        //     p.cur_span()
        //         .into_spanned(Expected(p.peek(), Token::Identifier))
        // })(self)
        // .map(|(ty, name)| Var::scalar(ty, name))
    }

    fn func_params(&mut self) -> Result<Vec<Var<'a>>> {
        self.delimited(
            Token::LeftParen,
            Self::sequence(Self::func_param, Token::Comma),
            Token::RightParen,
            |_| unreachable!(),
        )
    }

    fn void_function(&mut self) -> Result<Function<'a>> {
        self.preceded(
            Token::Void,
            |p| {
                p.cascade(
                    Self::ident,
                    |p| {
                        p.cascade(Self::func_params, Self::block, |p| {
                            p.cur_span().into_spanned(ExpectedBlock)
                        })
                    },
                    |p| p.expect(Token::LeftParen),
                )
            },
            |p| p.expect(Token::LeftParen),
        )
        .map(|(name, (args, body))| Function::new(name, args, body, None))
    }

    fn var_decl(&mut self, ty: Type) -> Result<Var<'a>> {
        self.ident()
            .and_then(|ident| self.opt_size().map(|size| Var::new(ty, ident, size)))
        // Self::cascade(Self::ident, Self::opt_size, |_| unreachable!())(self)
        //     .map(|(id, size)| Var::new(ty, id, size))
    }

    fn var_list(&mut self, ty: Type) -> Result<Vec<Var<'a>>> {
        Self::sequence(|p| p.var_decl(ty), Token::Comma)(self)
    }

    // FIXME: this can return outputs with wrong input **without reporting an error**
    fn field_or_function_decl(&mut self) -> Result<Or<Vec<Var<'a>>, Function<'a>>> {
        let vars_list = |p: &mut Self, ty, first| {
            p.bump();
            let vars = p.var_list(ty);
            match (vars, first) {
                (Parsed(mut vars), Parsed(first)) => Parsed({
                    vars.push(first);
                    vars
                }),
                (ErrorWithResult(mut vars), Parsed(first))
                | (ErrorWithResult(mut vars), ErrorWithResult(first))
                | (Parsed(mut vars), ErrorWithResult(first)) => ErrorWithResult({
                    vars.push(first);
                    vars
                }),
                (Nil, Parsed(first)) | (Error, Parsed(first)) => Parsed(vec![first]),
                (Nil, ErrorWithResult(first)) | (Error, ErrorWithResult(first)) => {
                    ErrorWithResult(vec![first])
                }
                (_, Nil) | (_, Error) => unreachable!(),
            }
        }.map(|vars| Or::First(vars));
        match self.peek() {
            Token::Void => self.void_function().map(Or::Second),
            Token::Int | Token::Bool => {
                let ty = self.var_type().unwrap_parsed();
                let id = self.ident();
                if id.has_output() {
                    match self.peek() {
                        // function declaration
                        Token::LeftParen => self
                            .cascade(Self::func_params, Self::block, |p| {
                                p.cur_span().into_spanned(ExpectedBlock)
                            })
                            .map(|(params, body)| {
                                Or::Second(Function::new(
                                    id.unwrap_output(),
                                    params,
                                    body,
                                    Some(ty),
                                ))
                            }),
                        // var declaration
                        Token::SquareLeft => {
                            let size = self.opt_size();
                            if !size.has_output() {
                                let error = self.expect(Token::DecimalLiteral);
                                (self.error_callback)(error);
                                return Error;
                            }
                            let first_var = if size.has_error() {
                                ErrorWithResult(Var::new(
                                    ty,
                                    id.unwrap_output(),
                                    size.unwrap_output(),
                                ))
                            } else {
                                Parsed(Var::new(ty, id.unwrap_output(), size.unwrap_output()))
                            };
                            let peek = self.peek();
                            if peek == Token::Comma {
                                vars_list(self, ty, first_var)
                            } else if peek == Token::Semicolon {
                                self.bump();
                                first_var.map(|var| Or::First(vec![var]))
                            } else {
                                first_var
                                    .map(|var| {
                                        let error = self.expect(Token::Semicolon);
                                        (self.error_callback)(error);
                                        Or::First(vec![var])
                                    })
                                    .into_parsed_error()
                            }
                        }
                        // var declaration
                        Token::Comma => {
                            let first_var = Parsed(Var::scalar(ty, id.unwrap_output()));
                            vars_list(self, ty, first_var)
                        }
                        // var declaration
                        Token::Semicolon => {
                            self.bump();
                            Parsed(Or::First(vec![Var::scalar(ty, id.unwrap_output())]))
                            // ErrorWithResult(Or::First(vec![Var::scalar(ty, id.unwrap_output())]))
                        }
                        // incomplete declaration
                        _ => {
                            let error = self.expect(Token::Semicolon);
                            (self.error_callback)(error);
                            ErrorWithResult(Or::First(vec![Var::scalar(ty, id.unwrap_output())]))
                        }
                    }
                } else {
                    Error
                }
            }
            _ => Nil,
        }
    }

    fn nested_expr(&mut self) -> Result<Expr<'a>> {
        self.span(|p| {
            p.delimited(Token::LeftParen, Self::expr, Token::RightParen, |p| {
                p.cur_span().into_spanned(ExpectedExpression)
            })
        })
        .map(|exp| exp.into_parts())
        .map(|(e, span)| Expr::nested(e, span))
    }

    fn int_literal(&mut self) -> Result<IntLiteral<'a>> {
        match self.peek() {
            Token::DecimalLiteral => Parsed(IntLiteral::from_decimal(self.bump().span())),
            Token::HexLiteral => Parsed(IntLiteral::from_hex(self.bump().span())),
            _ => Nil,
        }
    }

    fn bool_literal(&mut self) -> Result<BoolLiteral<'a>> {
        Self::exact_token(Token::True)(self)
            .map(|lit| BoolLiteral::from_spanned(lit.map(|_| true)))
            .or_else(|| {
                Self::exact_token(Token::False)(self)
                    .map(|lit| BoolLiteral::from_spanned(lit.map(|_| false)))
            })
    }

    fn char_literal(&mut self) -> Result<CharLiteral<'a>> {
        Self::exact_token(Token::CharLiteral(0))(self).map(|t| {
            CharLiteral::from_spanned(t.map(|tok| {
                if let Token::CharLiteral(c) = tok {
                    c
                } else {
                    unreachable!()
                }
            }))
        })
    }

    // fn eliteral(&mut self) -> Result<ELiteral<'a>> {
    //     self.int_literal()
    //         .map(|lit| ELiteral::int(lit))
    //         .or_else(|| self.bool_literal().map(|lit| ELiteral::bool(lit)))
    //         .or_else(|| self.char_literal().map(|lit| ELiteral::char(lit)))
    // }

    fn expr(&mut self) -> Result<Expr<'a>> {
        let expect_expression =
            |p: &mut Parser<'a, I, EH>| p.cur_span().into_spanned(ExpectedExpression);
        self.span(|p| {
            p.cascade(
                Self::or,
                |p| {
                    p.opt(|p| {
                        p.cascade(
                            |p| p.preceded(Token::Question, Self::expr, expect_expression),
                            |p| p.preceded(Token::Colon, Self::expr, expect_expression),
                            |p| p.cur_span().into_spanned(Expected(p.peek(), Token::Colon)),
                        )
                    })
                },
                |_| unreachable!(),
            )
        })
        .map(|spanned| spanned.into_parts())
        .map(|((e, opt_yes_no), span)| match opt_yes_no {
            Some((yes, no)) => Expr::ter(e, yes, no, span),
            None => e,
        })
    }

    fn fold<O1, O2>(
        mut parser: impl FnMut(&mut Parser<'a, I, EH>) -> Result<O1>,
        mut init: impl FnMut(&mut Parser<'a, I, EH>) -> Result<O2>,
        mut reducer: impl FnMut(O2, O1) -> O2,
    ) -> impl FnMut(&mut Parser<'a, I, EH>) -> Result<O2> {
        move |p| {
            let mut cur = init(p);
            if cur.has_output() {
                loop {
                    match parser(p) {
                        Nil => break cur,
                        Parsed(res) => {
                            cur = cur.map(|cur| reducer(cur, res));
                        }
                        Error => {
                            let res = cur.into_parsed_error();
                            break res;
                        }
                        ErrorWithResult(parsed) => {
                            cur = cur.into_parsed_error();
                            cur = cur.map(|cur| reducer(cur, parsed));
                        }
                    }
                }
            } else {
                cur
            }
        }
    }

    fn sequence<O1>(
        mut parser: impl FnMut(&mut Parser<'a, I, EH>) -> Result<O1>,
        separator: Token,
    ) -> impl FnMut(&mut Parser<'a, I, EH>) -> Result<Vec<O1>> {
        move |p| {
            let mut token_parser = Self::exact_token(separator);
            let mut parsed = Parsed(vec![]);
            loop {
                match parser(p) {
                    Parsed(res) => {
                        parsed = parsed.map(|mut v| {
                            v.push(res);
                            v
                        })
                    }
                    Nil => break parsed,
                    Error => break parsed.into_parsed_error(),
                    ErrorWithResult(res) => {
                        parsed = parsed
                            .map(|mut v| {
                                v.push(res);
                                v
                            })
                            .into_parsed_error()
                    }
                };
                match token_parser(p) {
                    Parsed(_) => (),
                    Nil => break parsed,
                    _ => unreachable!(),
                }
            }
        }
    }

    binop!(
        mul_div,
        unit_expr,
        Star => Mul,
        Slash => Div,
        Percent => Mod
    );

    binop!(
        add_sub,
        mul_div,
        Plus => Add,
        Minus => Sub
    );

    binop!(ord,
           add_sub,
           Greater => Greater,
           GreaterEqual => GreaterEqual,
           Less => Less,
           LessEqual => LessEqual);

    binop!(eq,
           ord,
           EqualEqual => Equal,
           NotEqual => NotEqual);

    binop!(and, eq, And => And,);

    binop!(or, and, Or => Or,);

    fn string_literal(&mut self) -> Result<StringLiteral<'a>> {
        Self::exact_token(Token::StringLiteral)(self)
            .map(|spanned| spanned.into_parts())
            .map(|(_, span)| StringLiteral::from_span(span))
    }

    fn call_arg(&mut self) -> Result<Arg<'a>> {
        self.expr()
            .map(Arg::from_expr)
            .or_else(|| self.string_literal().map(Arg::from_string))
    }

    fn call_args(&mut self) -> Result<Vec<Arg<'a>>> {
        let call_args_inner = |p: &mut Self| {
            let mut args = vec![];
            let mut comma = Self::exact_token(Token::Comma);
            let args = loop {
                let arg = p.call_arg();
                if !arg.has_output() {
                    break args;
                } else {
                    args.push(arg);
                    if comma(p).is_nil() {
                        break args;
                    } else {
                        continue;
                    }
                }
            };
            args.into_iter()
                .fold(Parsed(vec![]), |args, arg| match arg {
                    Parsed(arg) => args.map(|mut args| {
                        args.push(arg);
                        args
                    }),
                    ErrorWithResult(arg) => args.into_parsed_error().map(|mut args| {
                        args.push(arg);
                        args
                    }),
                    _ => unreachable!(),
                })
        };
        self.delimited(
            Token::LeftParen,
            |p| p.opt(call_args_inner),
            Token::RightParen,
            |_| unreachable!(),
        )
        .map(|args| args.unwrap_or_default())
    }

    fn eliteral(&mut self) -> Result<Expr<'a>> {
        self.int_literal()
            .map(|i| i.into())
            .or_else(|| self.char_literal().map(|c| c.into()))
            .or_else(|| self.bool_literal().map(|b| b.into()))
    }

    fn unit_expr(&mut self) -> Result<Expr<'a>> {
        self.len_expr()
            .or_else(|| self.eliteral())
            .or_else(|| self.neg())
            .or_else(|| self.not())
            .or_else(|| {
                self.call_or_loc().map(|c_or_lo| match c_or_lo {
                    Or::First(c) => c.into(),
                    Or::Second(lo) => lo.into(),
                })
            })
            .or_else(|| self.nested_expr())
    }

    fn block_elem(&mut self) -> Result<BlockElem<'a>> {
        self.field_or_function_decl()
            .map(|decl_or_func| match decl_or_func {
                Or::First(decl) => decl.into(),
                Or::Second(func) => func.into(),
            })
            .or_else(|| self.stmt().map(|stmt| stmt.into()))
    }

    fn block(&mut self) -> Result<Block<'a>> {
        use std::iter;
        if self.consume(Token::CurlyLeft).is_nil() {
            Nil
        } else {
            let block = iter::from_fn(|| match self.block_elem() {
                Parsed(elem) => Some(elem),
                Nil => None,
                ErrorWithResult(elem) => Some(elem),
                Error => {
                    panic!()
                }
            })
            .fold(BlockBuilder::new(), |mut builder, elem| {
                builder.add(elem);
                builder
            })
            .build();
            if self.consume(Token::CurlyRight).is_nil() {
                panic!()
            } else {
                Parsed(block)
            }
        }
    }

    /// parses if statements, allows parsing conditions that is not surrounded by `()`
    fn if_stmt(&mut self) -> Result<Stmt<'a>> {
        let expect_block = |p: &mut Self| p.cur_span().into_spanned(ExpectedBlock);
        self.cascade(
            |p| {
                p.preceded(
                    Token::If,
                    |p| p.cascade(Self::expr, Self::block, expect_block),
                    |p| p.cur_span().into_spanned(ExpectedExpression),
                )
            },
            |p| p.opt(|p| p.preceded(Token::Else, Self::block, expect_block)),
            |_| unreachable!(),
        )
        .map(|((cond, yes), no)| Stmt::if_stmt(cond, yes, no))
    }

    fn while_stmt(&mut self) -> Result<Stmt<'a>> {
        self.preceded(
            Token::While,
            |p| {
                p.cascade(Self::expr, Self::block, |p| {
                    p.cur_span().into_spanned(ExpectedBlock)
                })
            },
            |p| p.cur_span().into_spanned(ExpectedExpression),
        )
        .map(|(cond, body)| Stmt::while_stmt(cond, body))
    }

    fn return_stmt(&mut self) -> Result<Stmt<'a>> {
        self.delimited(
            Token::Return,
            |p| p.opt(Self::expr),
            Token::Semicolon,
            |_| unreachable!(),
        )
        .map(|expr| Stmt::return_stmt(expr))
    }

    fn break_stmt(&mut self) -> Result<Stmt<'a>> {
        self.terminated(Self::exact_token(Token::Break), Token::Semicolon)
            .map(|_| Stmt::break_stmt())
    }

    fn continue_stmt(&mut self) -> Result<Stmt<'a>> {
        self.terminated(Self::exact_token(Token::Continue), Token::Semicolon)
            .map(|_| Stmt::continue_stmt())
    }

    fn assign_expr(&mut self) -> Result<AssignExpr<'a>> {
        match self.peek() {
            Token::Increment => {
                self.bump();
                Parsed(AssignExpr::inc())
            }
            Token::Decrement => {
                self.bump();
                Parsed(AssignExpr::dec())
            }
            Token::Assign => {
                self.bump();
                self.expr().map(|expr| AssignExpr::assign(expr))
            }
            Token::AddAssign => {
                self.bump();
                self.expr().map(|expr| AssignExpr::add_assign(expr))
            }
            Token::SubAssign => {
                self.bump();
                self.expr().map(|expr| AssignExpr::sub_assign(expr))
            }
            _ => Nil,
        }
    }

    // fn for_stmt(&mut self) -> Result<Stmt<'a>> {
    // }

    fn call_or_assignment(&mut self) -> Result<Stmt<'a>> {
        self.call_or_loc()
            .and_then(|call_or_loc| match call_or_loc {
                Or::First(call) => Parsed(Stmt::call_stmt(call)),
                Or::Second(loc) => self
                    .assign_expr()
                    .map(|assignexpr| Assign::new(loc, assignexpr).into())
                    .map_nil(|| {
                        let error = self.cur_span().into_spanned(ExpectedAssignExpr);
                        (self.error_callback)(error);
                        Error
                    }),
            })
            .and_then(|stmt| {
                if self.peek() != Token::Semicolon {
                    let error = self
                        .cur_span()
                        .into_spanned(Expected(self.peek(), Token::Semicolon));
                    (self.error_callback)(error);
                    ErrorWithResult(stmt)
                } else {
                    self.bump();
                    Parsed(stmt)
                }
            })
    }

    fn loc(&mut self) -> Result<Loc<'a>> {
        let beg = self.cur_span();
        self.ident().and_then(|ident| {
            self.opt_index().map(|index| match index {
                Some(index) => Loc::with_offset(ident, index, beg.merge(self.cur_span())),
                None => Loc::from_ident(ident),
            })
        })
    }

    fn assign(&mut self) -> Result<Assign<'a>> {
        self.loc().and_then(|loc| {
            self.assign_expr()
                .map(|assignexpr| Assign::new(loc, assignexpr))
                .map_nil(|| {
                    let error = self.cur_span().into_spanned(ExpectedAssignExpr);
                    (self.error_callback)(error);
                    Error
                })
        })
    }

    fn consume(&mut self, token: Token) -> Result<()> {
        Self::exact_token(token)(self).map(|_| ())
    }

    fn for_inner_parens(&mut self) -> Result<(Assign<'a>, Expr<'a>, Assign<'a>)> {
        let for_error = |p: &mut Self| p.cur_span().into_spanned(Syn(Token::For));
        self.cascade(
            |p| {
                p.cascade(
                    Self::assign,
                    |p| p.preceded(Token::Semicolon, Self::expr, for_error),
                    for_error,
                )
            },
            |p| p.preceded(Token::Semicolon, Self::assign, for_error),
            for_error,
        )
        .map(|((init, cond), update)| (init, cond, update))
    }

    fn for_stmt(&mut self) -> Result<Stmt<'a>> {
        let for_error = |p: &mut Self| p.cur_span().into_spanned(Syn(Token::For));
        self.preceded(
            Token::For,
            |p| {
                p.cascade(
                    |p| {
                        p.delimited(
                            Token::LeftParen,
                            Self::for_inner_parens,
                            Token::RightParen,
                            for_error,
                        )
                    },
                    Self::block,
                    for_error,
                )
            },
            for_error,
        )
        .map(|((init, cond, update), body)| Stmt::for_stmt(init, cond, update, body))
    }

    fn stmt(&mut self) -> Result<Stmt<'a>> {
        self.if_stmt()
            .or_else(|| self.while_stmt())
            .or_else(|| self.return_stmt())
            .or_else(|| self.break_stmt())
            .or_else(|| self.continue_stmt())
            .or_else(|| self.call_or_assignment())
            .or_else(|| self.for_stmt())
    }

    fn doc_elem(&mut self) -> Result<DocElem<'a>> {
        self.field_or_function_decl()
            .map(|field_or_func| match field_or_func {
                Or::First(field) => DocElem::decl(field),
                Or::Second(func) => DocElem::function(func),
            })
            .or_else(|| self.import().map(|import| DocElem::import(import)))
    }

    pub fn doc_elems(mut self) -> impl Iterator<Item = DocElem<'a>> {
        use std::iter;
        iter::from_fn(move || {
            self.doc_elem()
                .map(|doc_elem| Some(doc_elem))
                .map_nil(|| Parsed(None))
                .unwrap_parsed()
        })
    }
}
