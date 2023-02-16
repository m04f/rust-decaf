use crate::lexer::Token::*;
use crate::parser::Error::*;
use crate::span::*;

macro_rules! span {
    ($span:ident, $text:expr) => {
        let span_source = SpanSource::new($text.as_bytes());
        let $span = span_source.source();
    };
}

use crate::lexer::tokens;
use crate::parser::Parser;
macro_rules! parser {
    ($parser:ident, $text:expr) => {
        println!("parsing: {}", $text);
        span!(text, $text);
        let mut $parser = Parser::new(
            tokens(text, |_| {}).map(|res| res.map(|res| res.unwrap())),
            |e| panic!("unexpected error: {e:?}"),
        );
    };
    ($parser:ident, $text:expr, $err:expr) => {
        println!("parsing: {}", $text);
        span!(text, $text);
        let mut $parser = Parser::new(
            tokens(text, |_| {}).map(|res| res.map(|res| res.unwrap())),
            $err,
        );
    };
}

#[test]
fn literals() {
    use crate::ast::*;
    parser!(parser, "1");
    assert_eq!(
        parser.expr().unwrap(),
        Expr::from(IntLiteral::from_decimal("1"))
    );

    parser!(parser, "0x1");
    assert_eq!(
        parser.expr().unwrap(),
        Expr::from(IntLiteral::from_hex("0x1"))
    );

    parser!(parser, "'c'");
    span!(expect, "'c'");
    let expect = expect.into_spanned(b'c');
    assert_eq!(
        parser.expr().unwrap(),
        Expr::from(CharLiteral::from_spanned(expect))
    );
    parser!(parser, "false");
    span!(expect, "false");
    let expect = expect.into_spanned(false);
    assert_eq!(
        parser.expr().unwrap(),
        Expr::from(BoolLiteral::from_spanned(expect))
    );
}

#[test]
fn bad_call_stmt_with_output() {
    parser!(parser, "call()", |err| assert_eq!(
        err.into_parts().0,
        Expected(Eof, Semicolon)
    ));
    let stmt = parser.stmt();
    println!("{:?}", stmt);
    assert!(parser.finised());
    assert!(stmt.is_ok());
    let mut errors = vec![
        ExpectedMatching(LeftParen, RightParen),
        Expected(Eof, Semicolon),
    ]
    .into_iter();
    parser!(parser, "call(", |err| {
        assert_eq!(err.into_parts().0, errors.next().unwrap());
    });
    let stmt = parser.stmt();
    assert!(parser.finised());
    assert!(stmt.is_ok());
}

#[test]
fn bool_decl() {
    parser!(parser, "bool a;");
    parser.field_or_function_decl().unwrap();
}

#[test]
fn multi_decl() {
    parser!(parser, r#" int a, b, c;"#);
    parser.field_or_function_decl().unwrap();
    assert!(parser.finised());
}

#[test]
fn not_expr() {
    use crate::ast::*;
    parser!(parser, "!0");
    let expr = parser.expr().unwrap();
    assert_eq!(expr, Expr::not(IntLiteral::from_decimal("0").into(), ""))
}

#[test]
fn neg_expr() {
    use crate::ast::*;
    parser!(parser, "-0");
    let expr = parser.expr().unwrap();
    assert_eq!(expr, Expr::neg(IntLiteral::from_decimal("0").into(), ""))
}

#[test]
fn mul_expr() {
    use crate::ast::*;
    parser!(parser, "a * b");
    let expr = parser.expr().unwrap();
    assert_eq!(
        expr,
        Expr::binop(
            Identifier::from_span("a").into(),
            Op::Mul,
            Identifier::from_span("b").into(),
            ""
        )
    );
}

#[test]
fn add_expr() {
    use crate::ast::*;
    parser!(parser, "a + b");
    let expr = parser.expr().unwrap();
    assert_eq!(
        expr,
        Expr::binop(
            Identifier::from_span("a").into(),
            Op::Add,
            Identifier::from_span("b").into(),
            ""
        )
    );
}

#[test]
fn basic_prec_expr() {
    use crate::ast::*;
    parser!(parser, "a + b * c");
    let expr = parser.expr().unwrap();
    assert_eq!(
        expr,
        Expr::binop(
            Identifier::from_span("a").into(),
            Op::Add,
            Expr::binop(
                Identifier::from_span("b").into(),
                Op::Mul,
                Identifier::from_span("c").into(),
                ""
            ),
            ""
        )
    );
}

#[test]
fn double_neg() {
    use crate::ast::*;
    parser!(parser, "- -0");
    let expr = parser.expr();
    let expr = expr.unwrap();
    assert_eq!(
        expr,
        Expr::neg(
            Expr::neg(IntLiteral::from_decimal("0").into(), ""),
            ""
        ),
    );
}

#[test]
fn multiple_params() {
    use crate::ast::*;
    parser!(parser, "(int b, int c)");
    let params = parser.func_params().unwrap();
    assert_eq!(
        params,
        vec![
            Var::scalar(Type::Int, "b".into()),
            Var::scalar(Type::Int, "c".into()),
        ]
    );
    assert!(parser.finised());
}

#[test]
fn index_expr() {
    use crate::ast::*;
    parser!(parser, "a[b+2]");
    let expr = parser.expr().unwrap();
    assert_eq!(
        expr,
        Expr::index(
            Identifier::from_span("a"),
            Expr::binop(
                Identifier::from_span("b").into(),
                Op::Add,
                IntLiteral::from_decimal("2").into(),
                ""
            ),
            ""
        )
    );
    assert!(parser.finised());
}

#[test]
fn no_semicolon_decl() {
    use super::Or;
    use crate::ast::*;
    let mut errors = vec![Expected(Eof, Semicolon)].into_iter();
    parser!(parser, "int a", |e| assert_eq!(
        *e.get(),
        errors.next().unwrap()
    ));
    let decl = parser.field_or_function_decl().unwrap();
    if let Or::First(var) = decl {
        assert_eq!(var, vec![Var::scalar(Type::Int, "a".into())]);
    } else {
        panic!()
    }
}

#[test]
fn decl_stmt_decl() {
    use crate::ast::Error::*;
    let mut err_count = 0;
    parser!(
        parser,
        "{ int a; a = a; int b; }",
        |e| if let Ast(MoveDeclTo { .. }) = e.get() {
            err_count += 1;
        } else {
            panic!()
        }
    );
    parser.block().unwrap();
    assert!(err_count == 1);
}
