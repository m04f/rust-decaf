use crate::lexer::Token::*;
use crate::parser::Error::*;
use crate::span::*;

macro_rules! parser {
    ($text:expr) => {{
        use crate::lexer::tokens;
        use crate::parser::Parser;
        println!("parsing: {}", $text);
        Parser::new(
            tokens($text.as_bytes(), |_| {}).map(|res| res.map(|res| res.unwrap())),
            |e| panic!("unexpected error: {e:?}"),
        )
    }};
    ($text:expr, $err:expr) => {{
        use crate::lexer::tokens;
        use crate::parser::Parser;
        println!("parsing: {}", $text);
        Parser::new(
            tokens($text.as_bytes(), |_| {}).map(|res| res.map(|res| res.unwrap())),
            $err,
        )
    }};
}

#[test]
fn literals() {
    use crate::ast::*;
    let mut parser = parser!("1");
    assert_eq!(
        parser.expr().unwrap(),
        Expr::from(IntLiteral::from_decimal("1"))
    );

    let mut parser = parser!("0x1");
    assert_eq!(
        parser.expr().unwrap(),
        Expr::from(IntLiteral::from_hex("0x1"))
    );

    let mut parser = parser!("'c'");
    assert_eq!(
        parser.expr().unwrap(),
        Expr::from(CharLiteral::from_spanned(
            Span::new(b"'c'").into_spanned(b'c')
        ))
    );
    let mut parser = parser!("false");
    assert_eq!(
        parser.expr().unwrap(),
        Expr::from(BoolLiteral::from_spanned(
            Span::new(b"false").into_spanned(false)
        ))
    );
}

#[test]
fn bad_call_stmt_with_output() {
    let mut parser = parser!("call()", |err| assert_eq!(
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
    let mut parser = parser!("call(", |err| {
        assert_eq!(err.into_parts().0, errors.next().unwrap());
    });
    let stmt = parser.stmt();
    assert!(parser.finised());
    assert!(stmt.is_ok());
}

#[test]
fn bool_decl() {
    let mut parser = parser!("bool a;");
    parser.field_or_function_decl().unwrap();
}

#[test]
fn multi_decl() {
    let mut parser = parser!(r#" int a, b, c;"#);
    parser.field_or_function_decl().unwrap();
    assert!(parser.finised());
}

#[test]
fn not_expr() {
    use crate::ast::*;
    let mut parser = parser!("!0");
    let expr = parser.expr().unwrap();
    assert_eq!(expr, Expr::not(IntLiteral::from_decimal("0").into(), ""))
}

#[test]
fn neg_expr() {
    use crate::ast::*;
    let mut parser = parser!("-0");
    let expr = parser.expr().unwrap();
    assert_eq!(expr, Expr::neg(IntLiteral::from_decimal("0").into(), ""))
}

#[test]
fn mul_expr() {
    use crate::ast::*;
    let mut parser = parser!("a * b");
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
    let mut parser = parser!("a + b");
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
    let mut parser = parser!("a + b * c");
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
    let mut parser = parser!("- -0");
    let expr = parser.expr();
    let expr = expr.unwrap();
    assert_eq!(
        expr,
        Expr::neg(
            Expr::neg(IntLiteral::from_decimal("0").into(), "").into(),
            ""
        ),
    );
}

#[test]
fn multiple_params() {
    use crate::ast::*;
    let mut parser = parser!("(int b, int c)");
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
    let mut parser = parser!("a[b+2]");
    let expr = parser.expr().unwrap();
    assert_eq!(
        expr,
        Expr::index(
            Identifier::from_span("a").into(),
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
    let mut parser = parser!("int a", |e| assert_eq!(*e.get(), errors.next().unwrap()));
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
    let mut parser = parser!(
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
