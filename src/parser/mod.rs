use crate::{lexer::Token, span::*};
use core::iter::Peekable;

mod error;
pub use error::*;
use Error::*;
pub mod ast;
use ast::checker::*;
use ast::*;

type Result<T> = std::result::Result<T, ExitStatus>;

/// the error returned by the parser.
#[derive(Debug, PartialEq, Eq)]
enum ExitStatus {
    /// clean means that the input did not change (relative to other parsers).
    Clean,
    /// dirty means that the input changed.
    Dirty,
}
use ExitStatus::*;

#[derive(Debug, PartialEq, Eq)]
enum Or<T1, T2> {
    First(T1),
    Second(T2),
}

#[derive(Debug)]
pub struct Parser<'a, I: Iterator<Item = Spanned<'a, Token>>, EH: FnMut(Error<'a>)> {
    tokens: Peekable<I>,
    error_callback: EH,
    last_pos: Span<'a>,
    error: bool,
}

macro_rules! binop {
    ($name:ident, $sub:ident, $first:ident => $first_mapped:ident, $($token:ident => $token_mapped:ident),*) => {
        fn $name(&mut self) -> Result<PExpr<'a>> {
            let op = move |p: &mut Parser<'a, I, EH>| {
                p.consume(Token::$first)
                    .map(|_| Op::$first_mapped)
                    $(.or_else(|_| p.consume(Token::$token).map(|_| Op::$token_mapped)))*
            };
            let mut expr = self.$sub()?;
            while let Ok(op) = op(self) {
                let rhs = self.$sub().map_err(|_| { self.expected_expression() })?;
                let span = expr.span().merge(rhs.span());
                expr = PExpr::new_binop(expr, rhs, op, span);
            }
            Ok(expr)
        }
    };
}

impl<'a, I: Iterator<Item = Spanned<'a, Token>>, EH: FnMut(Error<'a>)> Parser<'a, I, EH> {
    pub fn new(tokens: I, eh: EH) -> Self {
        let mut tokens = tokens.peekable();
        let beg = tokens.peek().unwrap().span().split_at(0).0;
        Self {
            tokens,
            error_callback: eh,
            last_pos: beg,
            error: false,
        }
    }

    fn peek(&mut self) -> Token {
        self.tokens.peek().map(|t| *t.get()).unwrap()
    }

    pub fn finised(&mut self) -> bool {
        self.peek() == Token::Eof
    }

    pub fn found_errors(&self) -> bool {
        self.error
    }

    fn expected_expression(&mut self) -> ExitStatus {
        let err = ExpectedExpression(self.cur_span());
        self.report_error(err);
        Dirty
    }

    fn expected_assignexpr(&mut self) -> ExitStatus {
        let err = ExpectedAssignExpr(self.cur_span());
        self.report_error(err);
        Dirty
    }

    fn expected_block(&mut self) -> ExitStatus {
        let err = ExpectedBlock(self.cur_span());
        self.report_error(err);
        Dirty
    }

    /// returns the next token and advances the iterator.
    /// panics if the iterator is empty.
    fn bump(&mut self) -> Spanned<'a, Token> {
        let poped = self.tokens.next().unwrap();
        let (tok, span) = poped.into_parts();
        let (span, last_pos) = span.split_at(span.len());
        self.last_pos = last_pos;
        span.into_spanned(tok)
    }

    fn ident(&mut self) -> Result<PIdentifier<'a>> {
        match self.peek() {
            Token::Identifier => Ok(self.bump().span().into()),
            _ => Err(Clean),
        }
    }

    fn cur_span(&mut self) -> Span<'a> {
        self.tokens.peek().unwrap().span()
    }

    fn report_error(&mut self, error: Error<'a>) {
        self.error = true;
        (self.error_callback)(error)
    }

    fn expected_token(&mut self, token: Token) -> Error<'a> {
        Expected {
            expected: token,
            found: self.peek(),
            span: self.cur_span(),
        }
    }

    fn start_span(&mut self) -> Span<'a> {
        self.cur_span()
    }

    fn end_span(&self, beg: Span<'a>) -> Span<'a> {
        beg.merge(self.last_pos)
    }

    fn import(&mut self) -> Result<PImport<'a>> {
        let beg = self.start_span();
        self.cur_span();
        self.consume(Token::Import)?;
        let identifier = self.ident().map_err(|_| {
            let error = self.expected_token(Token::Semicolon);
            self.report_error(error);
            Dirty
        })?;
        let span = self.end_span(beg);
        _ = self.consume(Token::Semicolon).map_err(|_| {
            let error = self.expected_token(Token::Semicolon);
            self.report_error(error)
        });
        Ok(span.into_spanned(identifier).into())
    }

    fn len_expr(&mut self) -> Result<PExpr<'a>> {
        let beg = self.start_span();
        self.consume(Token::Len)?;
        self.consume(Token::LeftParen).map_err(|_| {
            let error = self.expected_token(Token::LeftParen);
            self.report_error(error);
            Dirty
        })?;
        let ident = self.ident().map_err(|_| {
            let error = self.expected_token(Token::Identifier);
            self.report_error(error);
            Dirty
        })?;
        _ = self.consume(Token::RightParen).map_err(|_| {
            let error = self.expected_token(Token::RightParen);
            self.report_error(error);
        });
        let span = self.end_span(beg);
        Ok(PExpr::new_len(span.into_spanned(ident)))
    }

    fn neg(&mut self) -> Result<PExpr<'a>> {
        let beg = self.start_span();
        self.consume(Token::Minus)?;
        let expr = self.unit_expr().map_err(|_| self.expected_expression())?;
        let span = self.end_span(beg);
        Ok(PExpr::new_neg(span.into_spanned(expr)))
    }

    fn not(&mut self) -> Result<PExpr<'a>> {
        let beg = self.start_span();
        self.consume(Token::Not)?;
        let expr = self.unit_expr().map_err(|_| self.expected_expression())?;
        let span = self.end_span(beg);
        Ok(PExpr::new_not(span.into_spanned(expr)))
    }

    fn opt_index(&mut self) -> Result<Option<PExpr<'a>>> {
        if self.peek() == Token::SquareLeft {
            _ = self.consume(Token::SquareLeft);
            let expr = self.expr().map_err(|_| self.expected_expression())?;
            _ = self.consume(Token::SquareRight).map_err(|_| {
                let error = self.expected_token(Token::SquareRight);
                self.report_error(error)
            });
            Ok(Some(expr))
        } else {
            Ok(None)
        }
    }

    fn call_or_loc(&mut self) -> Result<Or<PCall<'a>, PLoc<'a>>> {
        let beg = self.start_span();
        let ident = self.ident()?;
        if self.peek() == Token::LeftParen {
            let args = self.call_args().map_err(|e| {
                // we do not report errors here since they should be reported by the args parser.
                // although we ensure that if there is an error it has to be dirty (we checked for
                // `(` before entering)
                assert_eq!(e, Dirty);
                Dirty
            })?;
            Ok(Or::First(PCall::new(ident, args, self.end_span(beg))))
        } else if self.peek() == Token::SquareLeft {
            let index = self.opt_index()?;
            Ok(Or::Second(PLoc::new(ident, index, self.end_span(beg))))
        } else {
            Ok(Or::Second(PLoc::new(ident, None, self.end_span(beg))))
        }
    }

    fn opt_size(&mut self) -> Result<Option<PIntLiteral<'a>>> {
        if self.peek() == Token::SquareLeft {
            self.bump();
            let lit = self.int_literal().map_err(|_| {
                let error = self.expected_token(Token::DecimalLiteral);
                self.report_error(error);
                Dirty
            })?;
            _ = self.consume(Token::SquareRight).map_err(|_| {
                let error = self.expected_token(Token::SquareRight);
                self.report_error(error)
            });
            let (lit, _) = lit.into_parts();
            Ok(Some(lit.try_into().unwrap()))
        } else {
            Ok(None)
        }
    }

    fn var_type(&mut self) -> Result<Type> {
        self.consume(Token::Int)
            .map(|_| Type::Int)
            .or(self.consume(Token::Bool).map(|_| Type::Bool))
    }

    fn func_param(&mut self) -> Result<PVar<'a>> {
        self.var_type().and_then(|ty| {
            self.ident()
                .map_err(|_| {
                    let error = self.expected_token(Token::Identifier);
                    self.report_error(error);
                    Dirty
                })
                .map(|ident| PVar::scalar(ty, ident))
        })
    }

    fn func_params(&mut self) -> Result<Vec<PVar<'a>>> {
        use std::iter;
        let left_paren_span = self.cur_span();
        self.consume(Token::LeftParen)?;
        let first_param = self.func_param();
        match first_param {
            Ok(param) => {
                let params = iter::once(param)
                    .chain(iter::from_fn(|| {
                        self.consume(Token::Comma).ok()?;
                        match self.func_param() {
                            Ok(res) => Some(res),
                            Err(_) => {
                                let error = Unexpected(Token::Comma, self.cur_span());
                                self.report_error(error);
                                None
                            }
                        }
                    }))
                    .collect::<Vec<_>>();
                _ = self.consume(Token::RightParen).map_err(|_| {
                    let error = ExpectedMatching {
                        left: Token::LeftParen,
                        lspan: left_paren_span,
                        right: Token::RightParen,
                        rspan: self.cur_span(),
                    };
                    self.report_error(error);
                });
                Ok(params)
            }
            Err(_) => {
                _ = self.consume(Token::RightParen).map_err(|_| {
                    let error = ExpectedMatching {
                        left: Token::LeftParen,
                        lspan: left_paren_span,
                        right: Token::RightParen,
                        rspan: self.cur_span(),
                    };
                    self.report_error(error);
                });
                Ok(vec![])
            }
        }
    }

    /// parses the parameters and body, (injects the parameters into the block).
    fn function_params_body(&mut self) -> Result<(Vec<PVar<'a>>, PBlock<'a>)> {
        let params = self.func_params().map_err(|_| {
            let error = self.expected_token(Token::LeftParen);
            self.report_error(error);
            Dirty
        })?;
        self.block()
            .map_err(|_| self.expected_block())
            .map(|body| (params, body))
    }

    fn void_function(&mut self) -> Result<PFunction<'a>> {
        let beg = self.start_span();
        self.consume(Token::Void)?;
        let name = self.ident().map_err(|_| {
            let error = self.expected_token(Token::Identifier);
            self.report_error(error);
            Dirty
        })?;
        self.function_params_body()
            .map(|(params, body)| PFunction::new(None, name, params, body, self.end_span(beg)))
    }

    fn var_decl(&mut self, ty: Type) -> Result<PVar<'a>> {
        let beg = self.start_span();
        self.ident().map(|ident| {
            self.opt_size()
                .map(|size| PVar::new(ty, ident, size, self.end_span(beg)))
                .unwrap_or_else(|_| PVar::scalar(ty, ident))
        })
    }

    fn var_list(&mut self, ty: Type) -> Result<Vec<PVar<'a>>> {
        use std::iter;
        self.var_decl(ty).map(|first| {
            iter::once(first)
                .chain(iter::from_fn(|| {
                    self.consume(Token::Comma).ok()?;
                    self.var_decl(ty)
                        .map_err(|_| {
                            let error = self.expected_token(Token::Identifier);
                            self.report_error(error);
                        })
                        .ok()
                }))
                .collect()
        })
    }

    fn field_or_function_decl(&mut self) -> Result<Or<Vec<PVar<'a>>, PFunction<'a>>> {
        let beg = self.start_span();
        let vars_after_comma = |p: &mut Self, ty, var: PVar<'a>| {
            p.bump();
            let vars = p
                .var_list(ty)
                .map(|mut vars| {
                    vars.push(var);
                    vars
                })
                .unwrap_or_else(|_| {
                    let error = p.expected_token(Token::Identifier);
                    p.report_error(error);
                    vec![var]
                });
            _ = p.consume(Token::Semicolon).map_err(|_| {
                let error = p.expected_token(Token::Semicolon);
                p.report_error(error);
            });

            Ok(Or::First(vars))
        };
        match self.peek() {
            Token::Void => self.void_function().map(Or::Second),
            Token::Int | Token::Bool => {
                let ty = self.var_type().unwrap();
                let ident = self.ident().map_err(|_| {
                    let error = self.expected_token(Token::Identifier);
                    self.report_error(error);
                    Dirty
                })?;
                match self.peek() {
                    Token::LeftParen => self
                        .function_params_body()
                        .map(|(params, body)| {
                            Or::Second(PFunction::new(
                                Some(ty),
                                ident,
                                params,
                                body,
                                self.end_span(beg),
                            ))
                        })
                        .map_err(|_| {
                            let error = self.expected_token(Token::LeftParen);
                            self.report_error(error);
                            Dirty
                        }),
                    Token::SquareLeft => {
                        let var = self
                            .opt_size()
                            .map(|size| PVar::new(ty, ident, size, self.end_span(ident.span())))
                            .unwrap_or_else(|_| PVar::scalar(ty, ident));
                        if self.peek() == Token::Comma {
                            vars_after_comma(self, ty, var)
                        } else {
                            _ = self.consume(Token::Semicolon).map_err(|_| {
                                let error = self.expected_token(Token::Semicolon);
                                self.report_error(error)
                            });
                            Ok(Or::First(vec![var]))
                        }
                    }
                    Token::Comma => vars_after_comma(self, ty, PVar::scalar(ty, ident)),
                    Token::Semicolon => {
                        self.bump();
                        Ok(Or::First(vec![PVar::scalar(ty, ident)]))
                    }
                    _ => {
                        let error = self.expected_token(Token::Semicolon);
                        self.report_error(error);
                        Ok(Or::First(vec![PVar::scalar(ty, ident)]))
                    }
                }
            }
            _ => Err(Clean),
        }
    }

    fn nested_expr(&mut self) -> Result<PExpr<'a>> {
        let beg = self.start_span();
        let left_paren_span = self.cur_span();
        self.consume(Token::LeftParen)?;
        let expr = self.expr().map_err(|_| self.expected_expression())?;
        _ = self.consume(Token::RightParen).map_err(|_| {
            let err = ExpectedMatching {
                lspan: left_paren_span,
                left: Token::LeftParen,
                right: Token::RightParen,
                rspan: self.cur_span(),
            };
            self.report_error(err)
        });
        Ok(PExpr::new_nested(self.end_span(beg).into_spanned(expr)))
    }

    fn int_literal(&mut self) -> Result<Spanned<'a, PLiteral<'a>>> {
        match self.peek() {
            Token::DecimalLiteral => Ok({
                let span = self.bump().span();
                span.into_spanned(PLiteral::decimal(span))
            }),
            Token::HexLiteral => Ok({
                let span = self.bump().span();
                span.into_spanned(PLiteral::hex(span.split_at(2).1))
            }),
            _ => Err(Clean),
        }
    }

    fn bool_literal(&mut self) -> Result<Spanned<'a, bool>> {
        match self.peek() {
            Token::True => Ok(self.bump().map(|_| true)),
            Token::False => Ok(self.bump().map(|_| false)),
            _ => Err(Clean),
        }
    }

    fn char_literal(&mut self) -> Result<Spanned<'a, u8>> {
        match self.peek() {
            Token::CharLiteral(c) => Ok(self.bump().map(|_| c)),
            _ => Err(Clean),
        }
    }

    fn expr(&mut self) -> Result<PExpr<'a>> {
        let beg = self.start_span();
        let e1 = self.or()?;
        match self.peek() {
            Token::Question => {
                self.bump();
                let yes = self.expr().map_err(|_| self.expected_expression())?;
                self.consume(Token::Colon).map_err(|_| {
                    let err = self.expected_token(Token::Colon);
                    self.report_error(err);
                    Dirty
                })?;
                let no = self.expr().map_err(|_| self.expected_expression())?;
                Ok(PExpr::new_ter(e1, yes, no, self.end_span(beg)))
            }
            _ => Ok(e1),
        }
    }

    // fn mul_div(&mut self) -> Result<Expr<'a>> {
    //     let op = move |p: &mut Parser<'a, I, EH>| {
    //         p.consume(Token::Star)
    //             .map(|_| Op::Mul)
    //             .or_else(|_| p.consume(Token::Slash).map(|_| Op::Div))
    //             .or_else(|_| p.consume(Token::Percent).map(|_| Op::Mod))
    //     };
    //     let mut expr = self.unit_expr()?;
    //     while let Ok(op) = op(self) {
    //         let rhs = self.unit_expr().map_err(|_| {
    //             self.report_error(ExpectedExpression);
    //             Dirty
    //         })?;
    //         let span = expr.span().merge(*rhs.span());
    //         expr = Expr::binop(expr, op, rhs, span);
    //     }
    //     Ok(expr)
    // }

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

    fn string_literal(&mut self) -> Result<PString<'a>> {
        match self.peek() {
            Token::StringLiteral => Ok(self.bump().span().into()),
            _ => Err(Clean),
        }
    }

    fn call_arg(&mut self) -> Result<PArg<'a>> {
        // NOTE: The order matters here since string_literal can not return a dirty error signal so
        // we start with it.
        self.string_literal()
            .map(|lit| lit.into())
            .or_else(|_| self.expr().map(|expr| expr.into()))
    }

    fn call_args(&mut self) -> Result<Vec<PArg<'a>>> {
        use std::iter;
        let left_paren_span = self.cur_span();
        self.consume(Token::LeftParen)?;
        let args = self
            .call_arg()
            .map(|first| {
                iter::once(first)
                    .chain(iter::from_fn(|| {
                        self.consume(Token::Comma).ok()?;
                        self.call_arg().ok()
                    }))
                    .collect()
            })
            .unwrap_or(vec![]);
        _ = self.consume(Token::RightParen).map_err(|_| {
            let err = ExpectedMatching {
                lspan: left_paren_span,
                left: Token::LeftParen,
                right: Token::RightParen,
                rspan: self.cur_span(),
            };
            self.report_error(err);
        });
        Ok(args)
    }

    fn eliteral(&mut self) -> Result<PExpr<'a>> {
        self.int_literal()
            .map(|i| i.into())
            .or_else(|_| self.char_literal().map(|c| c.into()))
            .or_else(|_| self.bool_literal().map(|b| b.into()))
    }

    fn unit_expr(&mut self) -> Result<PExpr<'a>> {
        self.len_expr()
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.eliteral()
                }
            })
            .or_else(|e| if e == Dirty { Err(Dirty) } else { self.neg() })
            .or_else(|e| if e == Dirty { Err(Dirty) } else { self.not() })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.call_or_loc().map(|c_or_lo| match c_or_lo {
                        Or::First(c) => c.into(),
                        Or::Second(lo) => lo.into(),
                    })
                }
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.nested_expr()
                }
            })
    }

    fn block_elem(&mut self) -> Result<PBlockElem<'a>> {
        let beg = self.start_span();
        self.field_or_function_decl()
            .map(|decl_or_func| match decl_or_func {
                Or::First(decls) => PBlockElem::decls(decls, self.end_span(beg)),
                Or::Second(func) => func.into(),
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.stmt().map(|stmt| stmt.into())
                }
            })
    }

    // FIXME: the block can terminate with a really messed up status.
    fn block(&mut self) -> Result<PBlock<'a>> {
        use std::iter;
        let left_bracket_span = self.cur_span();
        self.consume(Token::CurlyLeft)?;
        let mut block_checker = BlockChecker::new();
        Ok(iter::from_fn(|| {
            // if it returns an error then we did not finish the block yet so we can continue
            self.consume(Token::CurlyRight).err()?;
            match self.block_elem().map(|elem| {
                block_checker.check(&elem, |e| {
                    self.report_error(e);
                });
                elem
            }) {
                Ok(elem) => Some(elem),
                Err(_) => {
                    // there were no curly bracket and we could not parse anything. so we start to
                    // recover from this by consuming all tokens to the next curly brakcket.
                    let err = ExpectedMatching {
                        lspan: left_bracket_span,
                        left: Token::CurlyLeft,
                        right: Token::CurlyRight,
                        rspan: self.cur_span(),
                    };
                    self.report_error(err);
                    let mut depth = 0;
                    loop {
                        match self.peek() {
                            Token::CurlyLeft => depth += 1,
                            Token::CurlyRight => {
                                if depth == 0 {
                                    break;
                                } else {
                                    depth -= 1;
                                }
                            }
                            Token::Eof => break,
                            _ => {}
                        }
                        self.bump();
                    }
                    None
                }
            }
        })
        .fold(PBlock::new(), |mut block, elem| {
            block.add(elem);
            block
        }))
    }

    /// parses if statements, allows parsing conditions that is not surrounded by `()`
    fn if_stmt(&mut self) -> Result<PStmt<'a>> {
        let beg = self.start_span();
        self.consume(Token::If)?;
        let cond = self.expr().map_err(|_| {
            // FIXME: it is `(<expr>)` not an expression
            self.expected_expression()
        })?;
        let yes = self.block().map_err(|err| {
            assert_eq!(err, Clean);
            self.expected_block()
        })?;
        if self.consume(Token::Else).is_ok() {
            self.block()
                .map(|no| PStmt::r#if(cond, yes, Some(no), self.end_span(beg)))
                .map_err(|err| {
                    assert_eq!(err, Clean);
                    self.expected_block()
                })
        } else {
            Ok(PStmt::r#if(cond, yes, None, self.end_span(beg)))
        }
    }

    fn while_stmt(&mut self) -> Result<PStmt<'a>> {
        let beg = self.start_span();
        self.consume(Token::While)?;
        let cond = self.expr().map_err(|_| self.expected_expression())?;
        self.block()
            .map(|body| PStmt::r#while(cond, body, self.end_span(beg)))
            .map_err(|e| {
                assert_eq!(e, Clean);
                self.expected_block()
            })
    }

    fn return_stmt(&mut self) -> Result<PStmt<'a>> {
        let beg = self.start_span();
        self.consume(Token::Return)?;
        let expr = self
            .expr()
            .map(Some)
            .or_else(|e| if e == Dirty { Err(Dirty) } else { Ok(None) })?;
        self.consume(Token::Semicolon)
            .map(|_| PStmt::r#return(expr, self.end_span(beg)))
            .map_err(|_| {
                let error = self.expected_token(Token::Semicolon);
                self.report_error(error);
                Dirty
            })
    }

    fn break_stmt(&mut self) -> Result<PStmt<'a>> {
        let beg = self.start_span();
        self.consume(Token::Break)?;
        self.consume(Token::Semicolon)
            .map(|_| PStmt::r#break(self.end_span(beg)))
            .map_err(|_| {
                let error = self.expected_token(Token::Semicolon);
                self.report_error(error);
                Dirty
            })
    }

    fn continue_stmt(&mut self) -> Result<PStmt<'a>> {
        let beg = self.start_span();
        self.consume(Token::Continue)?;
        self.consume(Token::Semicolon)
            .map(|_| PStmt::r#continue(self.end_span(beg)))
            .map_err(|_| {
                let error = self.expected_token(Token::Semicolon);
                self.report_error(error);
                Dirty
            })
    }

    fn assign_expr(&mut self) -> Result<PAssignExpr<'a>> {
        match self.peek() {
            Token::Increment => {
                self.bump();
                Ok(PAssignExpr::inc())
            }
            Token::Decrement => {
                self.bump();
                Ok(PAssignExpr::dec())
            }
            Token::Assign => {
                self.bump();
                self.expr().map(PAssignExpr::assign)
            }
            Token::AddAssign => {
                self.bump();
                self.expr().map(PAssignExpr::add_assign)
            }
            Token::SubAssign => {
                self.bump();
                self.expr().map(PAssignExpr::sub_assign)
            }
            _ => Err(Clean),
        }
    }

    fn call_or_assignment(&mut self) -> Result<PStmt<'a>> {
        let beg = self.start_span();
        let stmt = self
            .call_or_loc()
            .and_then(|call_or_loc| match call_or_loc {
                Or::First(call) => Ok(call.into()),
                Or::Second(loc) => self
                    .assign_expr()
                    .map(|assignexpr| PAssign::new(loc, assignexpr, self.end_span(beg)).into())
                    .map_err(|_| self.expected_assignexpr()),
            })?;
        _ = self.consume(Token::Semicolon).map_err(|_| {
            let error = self.expected_token(Token::Semicolon);
            self.report_error(error)
        });
        Ok(stmt)
    }

    fn loc(&mut self) -> Result<PLoc<'a>> {
        let beg = self.cur_span();
        self.ident().and_then(|ident| {
            self.opt_index()
                .map(|index| PLoc::new(ident, index, beg.merge(self.cur_span())))
        })
    }

    fn assign(&mut self) -> Result<PAssign<'a>> {
        let beg = self.start_span();
        self.loc().and_then(|loc| {
            self.assign_expr()
                .map(|assignexpr| PAssign::new(loc, assignexpr, self.end_span(beg)))
                .map_err(|_| self.expected_assignexpr())
        })
    }

    fn consume(&mut self, token: Token) -> Result<()> {
        match self.peek() {
            Token::CharLiteral(_) if let Token::CharLiteral(_) = token => {
                self.bump();
                Ok(())
            }
            tok if tok == token => {
                self.bump();
                Ok(())
            }
            _ => Err(Clean)
        }
    }

    fn for_inner_parens(&mut self) -> Result<(PAssign<'a>, PExpr<'a>, PAssign<'a>)> {
        let assign = self.assign()?;
        _ = self.consume(Token::Semicolon).map_err(|_| {
            let error = self.expected_token(Token::Semicolon);
            self.report_error(error);
        });
        let expr = self.expr().map_err(|_| self.expected_expression())?;
        _ = self.consume(Token::Semicolon).map_err(|_| {
            let error = self.expected_token(Token::Semicolon);
            self.report_error(error);
        });
        let update = self.assign().map_err(|_| self.expected_assignexpr())?;
        _ = self.consume(Token::RightParen).map_err(|_| {
            let error = self.expected_token(Token::RightParen);
            self.report_error(error);
        });
        Ok((assign, expr, update))
    }

    fn for_stmt(&mut self) -> Result<PStmt<'a>> {
        let beg = self.start_span();
        self.consume(Token::For)?;
        self.consume(Token::LeftParen).map_err(|_| {
            let error = self.expected_token(Token::LeftParen);
            self.report_error(error);
            Dirty
        })?;
        let (init, cond, update) = self
            .for_inner_parens()
            .map_err(|_| self.expected_assignexpr())?;
        let body = self.block().map_err(|_| self.expected_block())?;
        Ok(PStmt::r#for(init, cond, update, body, self.end_span(beg)))
    }

    fn stmt(&mut self) -> Result<PStmt<'a>> {
        self.if_stmt()
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.while_stmt()
                }
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.return_stmt()
                }
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.break_stmt()
                }
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.continue_stmt()
                }
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.call_or_assignment()
                }
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.for_stmt()
                }
            })
            .map(|stmt| {
                StmtChecker::check(&stmt, |e| self.report_error(e));
                stmt
            })
    }

    fn doc_elem(&mut self) -> Result<PDocElem<'a>> {
        let beg = self.start_span();
        self.field_or_function_decl()
            .map(|field_or_func| match field_or_func {
                Or::First(field) => PDocElem::decls(field, self.end_span(beg)),
                Or::Second(func) => func.into(),
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.import().map(|import| import.into())
                }
            })
    }

    pub fn doc_elems(&mut self) -> impl Iterator<Item = PDocElem<'a>> + '_ {
        use std::iter;
        let mut elem_checker = RootChecker::new();
        iter::from_fn(move || {
            self.doc_elem()
                .map(|doc_elem| {
                    elem_checker.check(&doc_elem, |e| self.report_error(e));
                    Some(doc_elem)
                })
                .unwrap_or_else(|_| None)
        })
    }
}
