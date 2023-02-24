use crate::ast::{
    self, ELiteral, Error as AstError, IntLiteral, Op, RootChecker, StmtChecker, Type,
};

use crate::span::*;

use crate::lexer::Token;

use ExitStatus::*;
type Result<T> = std::result::Result<T, ExitStatus>;

use Error::*;

use core::iter::Peekable;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error<'a> {
    Expected(Token, Token),
    ExpectedExpression,
    ExpectedBlock,
    LenNoArg,
    ExpectedMatching(Token, Token),
    ExpectedAssignExpr,
    Unexpected(Token),
    Ast(AstError<'a>),
}

/// the error returned by the parser.
#[derive(Debug, PartialEq, Eq)]
enum ExitStatus {
    /// clean means that the input did not change (relative to other parsers).
    Clean,
    /// dirty means that the input changed.
    Dirty,
}

#[derive(Debug, PartialEq, Eq)]
enum Or<T1, T2> {
    First(T1),
    Second(T2),
}

#[derive(Debug)]
pub struct Parser<'a, I: Iterator<Item = Spanned<'a, Token>>, EH: FnMut(Spanned<'a, Error>)> {
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
                let rhs = self.$sub().map_err(|_| {
                    self.report_error(ExpectedExpression);
                    Dirty
                })?;
                let span = expr.span().merge(*rhs.span());
                expr = PExpr::binop(expr, op, rhs, span);
            }
            Ok(expr)
        }
    };
}

#[derive(Debug, Clone)]
pub enum PArg<'a> {
    String(ast::StringLiteral<Span<'a>>),
    Expr(PExpr<'a>),
}

impl<'a> PBlock<'a> {
    pub fn add(&mut self, elem: PBlockElem<'a>) {
        match elem {
            PBlockElem::Func(_) => {}
            PBlockElem::Decl { decls, .. } => self.decls.extend(decls),
            PBlockElem::Stmt(stmt) => self.stmts.push(stmt),
        }
    }
}

pub type PExpr<'a> = ast::Expr<PELiteral<'a>, PArg<'a>, Span<'a>, ()>;
pub type PStmt<'a> = ast::Stmt<PELiteral<'a>, PArg<'a>, Span<'a>, Vec<PVar<'a>>, ()>;
pub type PBlock<'a> = ast::Block<PELiteral<'a>, PArg<'a>, Span<'a>, Vec<PVar<'a>>, ()>;
pub type PBlockElem<'a> = ast::BlockElem<Span<'a>, PBlock<'a>, PArgs<'a>, PStmt<'a>>;
pub type PIdentifier<'a> = ast::Identifier<Span<'a>>;
pub type PImport<'a> = ast::Import<Span<'a>>;
pub type PIntLiteral<'a> = ast::IntLiteral<Span<'a>>;
pub type PELiteral<'a> = ast::ELiteral<Span<'a>>;
pub type PBoolLiteral<'a> = ast::BoolLiteral<Span<'a>>;
pub type PCharLiteral<'a> = ast::CharLiteral<Span<'a>>;
pub type PStringLiteral<'a> = ast::StringLiteral<Span<'a>>;
pub type PArgs<'a> = Vec<PVar<'a>>;
pub type PFunction<'a> = ast::Function<PBlock<'a>, PArgs<'a>, Span<'a>>;
pub type PVar<'a> = ast::Var<Span<'a>>;
pub type PCall<'a> = ast::Call<Span<'a>, PArg<'a>>;
pub type PLoc<'a> = ast::Loc<PELiteral<'a>, PArg<'a>, Span<'a>, ()>;
pub type PAssign<'a> = ast::Assign<PELiteral<'a>, PArg<'a>, Span<'a>, ()>;
pub type PAssignExpr<'a> = ast::AssignExpr<PELiteral<'a>, PArg<'a>, Span<'a>, ()>;
pub type PDocElem<'a> = ast::DocElem<PBlock<'a>, PArgs<'a>, Span<'a>>;
pub type PBlockChecker<'a> = ast::BlockChecker<'a>;
pub type PRoot<'a> = ast::Root<Span<'a>, Vec<PVar<'a>>, Vec<PFunction<'a>>>;

impl<'a> PArg<'a> {
    pub fn from_expr(expr: PExpr<'a>) -> PArg {
        PArg::Expr(expr)
    }

    pub fn from_string(lit: ast::StringLiteral<Span<'a>>) -> Self {
        PArg::String(lit)
    }

    pub fn span(&self) -> &Span<'a> {
        match self {
            Self::String(lit) => lit.span(),
            Self::Expr(expr) => expr.span(),
        }
    }
}

impl<'a> From<PStmt<'a>> for PBlockElem<'a> {
    fn from(value: PStmt<'a>) -> Self {
        PBlockElem::Stmt(value)
    }
}

impl<'a> From<PFunction<'a>> for PBlockElem<'a> {
    fn from(value: PFunction<'a>) -> Self {
        PBlockElem::Func(value)
    }
}

impl<'a> From<(Vec<PVar<'a>>, Span<'a>)> for PBlockElem<'a> {
    fn from((decls, span): (Vec<PVar<'a>>, Span<'a>)) -> Self {
        PBlockElem::Decl { span, decls }
    }
}

impl<'a> From<PCall<'a>> for PExpr<'a> {
    fn from(value: PCall<'a>) -> Self {
        PExpr {
            extra: (),
            inner: ast::ExprInner::Call(value),
        }
    }
}

impl<'a> From<Spanned<'a, u8>> for PExpr<'a> {
    fn from(value: Spanned<'a, u8>) -> Self {
        ast::ExprInner::Literal {
            span: value.span(),
            value: PELiteral::Char(*value.get()),
        }
        .into()
    }
}

impl<'a> From<Spanned<'a, bool>> for PExpr<'a> {
    fn from(value: Spanned<'a, bool>) -> Self {
        ast::ExprInner::Literal {
            span: value.span(),
            value: PELiteral::Bool(*value.get()),
        }
        .into()
    }
}

impl<'a> From<Spanned<'a, PELiteral<'a>>> for PExpr<'a> {
    fn from(value: Spanned<'a, PELiteral<'a>>) -> Self {
        let (value, span) = value.into_parts();
        ast::ExprInner::Literal { span, value }.into()
    }
}

impl<'a> From<PELiteral<'a>> for PIntLiteral<'a> {
    fn from(value: PELiteral<'a>) -> Self {
        match value {
            PELiteral::Decimal(i) => PIntLiteral::Decimal(i),
            PELiteral::Hex(i) => PIntLiteral::Hex(i),
            _ => panic!(),
        }
    }
}

impl<'a, I: Iterator<Item = Spanned<'a, Token>>, EH: FnMut(Spanned<'a, Error>)> Parser<'a, I, EH> {
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
        // println!("{:?}", self.peek());
        self.peek() == Token::Eof
    }

    pub fn found_errors(&self) -> bool {
        self.error
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
            Token::Identifier => Ok(PIdentifier::from_span(self.bump().span())),
            _ => Err(Clean),
        }
    }

    fn cur_span(&mut self) -> Span<'a> {
        self.tokens.peek().unwrap().span()
    }

    fn report_error(&mut self, error: Error) {
        self.error = true;
        let error = self.cur_span().into_spanned(error);
        (self.error_callback)(error)
    }

    fn expected_token(&mut self, token: Token) -> Error<'a> {
        Expected(self.peek(), token)
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
        Ok(PImport::from_ident(span.into_spanned(identifier)))
    }

    fn exact(&mut self, token: Token) -> Result<Spanned<'a, Token>> {
        if self.peek() == token {
            Ok(self.bump())
        } else {
            Err(Clean)
        }
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
        Ok(PExpr::len(ident, span))
    }

    fn neg(&mut self) -> Result<PExpr<'a>> {
        let beg = self.start_span();
        self.consume(Token::Minus)?;
        let expr = self.unit_expr().map_err(|_| {
            self.report_error(ExpectedExpression);
            Dirty
        })?;
        let span = self.end_span(beg);
        Ok(PExpr::neg(expr, span))
    }

    fn not(&mut self) -> Result<PExpr<'a>> {
        let beg = self.start_span();
        self.consume(Token::Not)?;
        let expr = self.unit_expr().map_err(|_| {
            self.report_error(ExpectedExpression);
            Dirty
        })?;
        let span = self.end_span(beg);
        Ok(PExpr::not(expr, span))
    }

    fn opt_index(&mut self) -> Result<Option<PExpr<'a>>> {
        if self.peek() == Token::SquareLeft {
            _ = self.consume(Token::SquareLeft);
            let expr = self.expr().map_err(|_| {
                self.report_error(ExpectedExpression);
                Dirty
            })?;
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
            let index = self.opt_index();
            if let Ok(index) = index {
                Ok(Or::Second(PLoc::with_offset(
                    ident,
                    index.unwrap(),
                    self.end_span(beg),
                )))
            } else {
                Ok(Or::Second(PLoc::from_ident(ident)))
            }
        } else {
            Ok(Or::Second(PLoc::from_ident(ident)))
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
            Ok(Some(lit.into()))
        } else {
            Ok(None)
        }
    }

    fn var_type(&mut self) -> Result<Type> {
        self.consume(Token::Int)
            .map(|_| Type::int_type())
            .or(self.consume(Token::Bool).map(|_| Type::bool_type()))
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
                                let error = Unexpected(Token::Comma);
                                self.report_error(error);
                                None
                            }
                        }
                    }))
                    .collect::<Vec<_>>();
                _ = self.consume(Token::RightParen).map_err(|_| {
                    let error = ExpectedMatching(Token::LeftParen, Token::RightParen);
                    self.report_error(error);
                });
                Ok(params)
            }
            Err(_) => {
                _ = self.consume(Token::RightParen).map_err(|_| {
                    let error = ExpectedMatching(Token::LeftParen, Token::RightParen);
                    self.report_error(error);
                });
                Ok(vec![])
            }
        }
    }

    /// parses the parameters and body, (injects the parameters into the block).
    fn function_params_body(&mut self) -> Result<(PArgs<'a>, PBlock<'a>)> {
        let params = self.func_params().map_err(|_| {
            let error = self.expected_token(Token::LeftParen);
            self.report_error(error);
            Dirty
        })?;
        self.block()
            .map_err(|_| {
                self.report_error(ExpectedBlock);
                Dirty
            })
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
            .map(|(params, body)| PFunction::new(name, body, params, None, self.end_span(beg)))
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
                    vars.push(var.clone());
                    vars
                })
                .unwrap_or_else(|_| {
                    let error = p.expected_token(Token::Identifier);
                    p.report_error(error);
                    vec![var.clone()]
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
                                ident,
                                body,
                                params,
                                Some(ty),
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
                            .map(|size| {
                                PVar::array(ty, ident, size.unwrap(), self.end_span(*ident.span()))
                            })
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
        self.consume(Token::LeftParen)?;
        let expr = self.expr().map_err(|_| {
            self.report_error(ExpectedExpression);
            Dirty
        })?;
        _ = self
            .consume(Token::RightParen)
            .map_err(|_| self.report_error(ExpectedMatching(Token::LeftParen, Token::RightParen)));
        Ok(PExpr::nested(expr, self.end_span(beg)))
    }

    fn int_literal(&mut self) -> Result<Spanned<'a, PELiteral<'a>>> {
        match self.peek() {
            Token::DecimalLiteral => Ok({
                let span = self.bump().span();
                span.into_spanned(PELiteral::decimal(span))
            }),
            Token::HexLiteral => Ok({
                let span = self.bump().span();
                span.into_spanned(PELiteral::hex(span.split_at(2).1))
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
                let yes = self.expr().map_err(|_| {
                    self.report_error(ExpectedExpression);
                    Dirty
                })?;
                self.consume(Token::Colon).map_err(|_| {
                    self.report_error(ExpectedMatching(Token::Question, Token::Colon));
                    Dirty
                })?;
                let no = self.expr().map_err(|_| {
                    self.report_error(ExpectedExpression);
                    Dirty
                })?;
                Ok(PExpr::ter(e1, yes, no, self.end_span(beg)))
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

    fn string_literal(&mut self) -> Result<PStringLiteral<'a>> {
        match self.peek() {
            Token::StringLiteral => Ok(PStringLiteral::from_span(self.bump().span())),
            _ => Err(Clean),
        }
    }

    fn call_arg(&mut self) -> Result<PArg<'a>> {
        // NOTE: The order matters here since string_literal can not return a dirty error signal so
        // we start with it.
        self.string_literal()
            .map(PArg::from_string)
            .or_else(|_| self.expr().map(PArg::from_expr))
    }

    fn call_args(&mut self) -> Result<Vec<PArg<'a>>> {
        use std::iter;
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
            let error = ExpectedMatching(Token::LeftParen, Token::RightParen);
            self.report_error(error);
        });
        Ok(args)
    }

    fn eliteral(&mut self) -> Result<PExpr<'a>> {
        self.int_literal()
            .map(|i| i.into())
            .or_else(|e| {
                assert_eq!(e, Clean);
                self.char_literal().map(|c| c.into())
            })
            .or_else(|e| {
                assert_eq!(e, Clean);
                self.bool_literal().map(|b| b.into())
            })
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
                Or::First(decl) => (decl, self.end_span(beg)).into(),
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
        self.consume(Token::CurlyLeft)?;
        let mut block_checker = PBlockChecker::new();
        Ok(iter::from_fn(|| {
            // if it returns an error then we did not finish the block yet so we can continue
            self.consume(Token::CurlyRight).err()?;
            match self.block_elem().map(|elem| {
                block_checker.check(&elem, |e| {
                    self.report_error(Ast(e));
                });
                elem
            }) {
                Ok(elem) => Some(elem),
                Err(_) => {
                    // there were no curly bracket and we could not parse anything. so we start to
                    // recover from this by consuming all tokens to the next curly brakcket.
                    self.report_error(ExpectedMatching(Token::CurlyLeft, Token::CurlyRight));
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
            self.report_error(ExpectedExpression);
            Dirty
        })?;
        let yes = self.block().map_err(|err| {
            assert_eq!(err, Clean);
            self.report_error(ExpectedBlock);
            Dirty
        })?;
        if self.consume(Token::Else).is_ok() {
            self.block()
                .map(|no| PStmt::if_stmt(cond, yes, Some(no), self.end_span(beg)))
                .map_err(|err| {
                    assert_eq!(err, Clean);
                    self.report_error(ExpectedBlock);
                    Dirty
                })
        } else {
            Ok(PStmt::if_stmt(cond, yes, None, self.end_span(beg)))
        }
    }

    fn while_stmt(&mut self) -> Result<PStmt<'a>> {
        let beg = self.start_span();
        self.consume(Token::While)?;
        let cond = self.expr().map_err(|_| {
            self.report_error(ExpectedExpression);
            Dirty
        })?;
        self.block()
            .map(|body| PStmt::while_stmt(cond, body, self.end_span(beg)))
            .map_err(|e| {
                assert_eq!(e, Clean);
                self.report_error(ExpectedBlock);
                Dirty
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
            .map(|_| PStmt::return_stmt(expr, self.end_span(beg)))
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
            .map(|_| PStmt::break_stmt(self.end_span(beg)))
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
            .map(|_| PStmt::continue_stmt(self.end_span(beg)))
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
                Or::First(call) => Ok(PStmt::call_stmt(call)),
                Or::Second(loc) => self
                    .assign_expr()
                    .map(|assignexpr| PAssign::new(loc, assignexpr, self.end_span(beg)).into())
                    .map_err(|_| {
                        self.report_error(ExpectedAssignExpr);
                        Dirty
                    }),
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
            self.opt_index().map(|index| match index {
                Some(index) => PLoc::with_offset(ident, index, beg.merge(self.cur_span())),
                None => PLoc::from_ident(ident),
            })
        })
    }

    fn assign(&mut self) -> Result<PAssign<'a>> {
        let beg = self.start_span();
        self.loc().and_then(|loc| {
            self.assign_expr()
                .map(|assignexpr| PAssign::new(loc, assignexpr, self.end_span(beg)))
                .map_err(|_| {
                    self.report_error(ExpectedAssignExpr);
                    Dirty
                })
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

    fn consume_or_dirty(&mut self, token: Token, on_err: impl FnOnce(&mut Self)) -> Result<()> {
        self.consume(token).map_err(|_| {
            on_err(self);
            Dirty
        })
    }

    fn for_inner_parens(&mut self) -> Result<(PAssign<'a>, PExpr<'a>, PAssign<'a>)> {
        let assign = self.assign()?;
        _ = self.consume(Token::Semicolon).map_err(|_| {
            let error = self.expected_token(Token::Semicolon);
            self.report_error(error);
        });
        let expr = self.expr().map_err(|_| {
            self.report_error(ExpectedExpression);
            Dirty
        })?;
        _ = self.consume(Token::Semicolon).map_err(|_| {
            let error = self.expected_token(Token::Semicolon);
            self.report_error(error);
        });
        let update = self.assign().map_err(|_| {
            self.report_error(ExpectedAssignExpr);
            Dirty
        })?;
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
        let (init, cond, update) = self.for_inner_parens().map_err(|_| {
            self.report_error(ExpectedAssignExpr);
            Dirty
        })?;
        let body = self.block().map_err(|_| {
            self.report_error(ExpectedBlock);
            Dirty
        })?;
        Ok(PStmt::for_stmt(
            init,
            cond,
            update,
            body,
            self.end_span(beg),
        ))
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
                StmtChecker::check(&stmt, |e| self.report_error(Ast(e)));
                stmt
            })
    }

    fn doc_elem(&mut self) -> Result<PDocElem<'a>> {
        let beg = self.start_span();
        self.field_or_function_decl()
            .map(|field_or_func| match field_or_func {
                Or::First(field) => PDocElem::decl(field, self.end_span(beg)),
                Or::Second(func) => PDocElem::function(func),
            })
            .or_else(|e| {
                if e == Dirty {
                    Err(Dirty)
                } else {
                    self.import().map(PDocElem::import)
                }
            })
    }

    pub fn doc_elems(&mut self) -> impl Iterator<Item = PDocElem<'a>> + '_ {
        use std::iter;
        let mut elem_checker = RootChecker::new();
        iter::from_fn(move || {
            self.doc_elem()
                .map(|doc_elem| {
                    elem_checker.check(&doc_elem, |e| self.report_error(Ast(e)));
                    Some(doc_elem)
                })
                .unwrap_or_else(|_| None)
        })
    }
}

impl<'a> FromIterator<PDocElem<'a>> for PRoot<'a> {
    fn from_iter<T: IntoIterator<Item = PDocElem<'a>>>(iter: T) -> Self {
        iter.into_iter().fold(PRoot::default(), |mut root, elem| {
            match elem {
                PDocElem::Decl(decls, _) => root.decls.extend(decls),
                PDocElem::Function(func) => root.funcs.push(func),
                PDocElem::Import(import) => root.imports.push(import),
            };
            root
        })
    }
}

use crate::ast::ExprInner;

impl<'a, Arg> ExprInner<PELiteral<'a>, Arg, Span<'a>, ()> {
    pub fn int_literal(self) -> Option<IntLiteral<Span<'a>>> {
        match self {
            ExprInner::Literal {
                value: ELiteral::Decimal(lit),
                ..
            } => Some(IntLiteral::Decimal(lit)),
            ExprInner::Literal {
                value: ELiteral::Hex(lit),
                ..
            } => Some(IntLiteral::Hex(lit)),
            _ => None,
        }
    }
}

impl<'a> PExpr<'a> {
    pub fn is_intliteral(&self) -> bool {
        matches!(
            self.inner,
            ast::ExprInner::Literal {
                value: ELiteral::Decimal(_) | ELiteral::Hex(_),
                ..
            }
        )
    }

    pub fn literal(self) -> Option<PELiteral<'a>> {
        match self.inner {
            ExprInner::Literal { value, .. } => Some(value),
            _ => None,
        }
    }
}
