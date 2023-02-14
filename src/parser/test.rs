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

#[cfg(test)]
mod arb_ast {
    use crate::ast::{ELiteral, Expr, Identifier, IntLiteral, Op};
    use proptest::prelude::*;
    use std::string::ToString;

    fn int_literal() -> impl Strategy<Value = IntLiteral<String>> {
        prop_oneof![
            proptest::string::string_regex("[0-9]+")
                .unwrap()
                .prop_map(|num| IntLiteral::from_decimal(num)),
            proptest::string::string_regex("0x[0-9a-fA-F]+")
                .unwrap()
                .prop_map(|num| IntLiteral::from_hex(num)),
        ]
    }

    fn ident() -> impl Strategy<Value = Identifier<String>> {
        // to avoid collisions with keywords we use uppercase chars only
        proptest::string::string_regex("[A-Z_][A-Z0-9_]*")
            .unwrap()
            .prop_map(|ident| Identifier::from_span(ident))
    }

    fn expr() -> impl Strategy<Value = Expr<String>> {
        let leaf = prop_oneof![
            int_literal().prop_map(|lit| Expr::literal(ELiteral::int(lit))),
            ident().prop_map(|ident| Expr::ident(ident)),
        ];
        leaf.prop_recursive(5, 15, 5, |inner| {
            prop_oneof![
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Add,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Sub,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Mul,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Div,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Mod,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Equal,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::NotEqual,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Less,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Greater,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::LessEqual,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::GreaterEqual,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::And,
                    b,
                    "".to_string()
                )),
                (inner.clone(), inner.clone()).prop_map(|(a, b)| Expr::binop(
                    a,
                    Op::Or,
                    b,
                    "".to_string()
                )),
                inner
                    .clone()
                    .prop_map(|expr| Expr::neg(expr, "".to_string())),
                inner
                    .clone()
                    .prop_map(|expr| Expr::not(expr, "".to_string())),
                inner
                    .clone()
                    .prop_map(|expr| Expr::nested(expr, "".to_string())),
                inner
            ]
        })
    }

    impl ToString for IntLiteral<String> {
        fn to_string(&self) -> String {
            use IntLiteral::*;
            match self {
                Decimal(num) => num.to_string(),
                Hex(num) => num.to_string(),
            }
        }
    }

    impl ToString for ELiteral<String> {
        fn to_string(&self) -> String {
            use ELiteral::*;
            match self {
                Int(lit) => lit.to_string(),
                _ => unreachable!(),
            }
        }
    }

    impl ToString for Identifier<String> {
        fn to_string(&self) -> String {
            self.span().to_string()
        }
    }

    impl ToString for Op {
        fn to_string(&self) -> String {
            use Op::*;
            match self {
                Add => "+",
                Sub => "-",
                Mul => "*",
                Div => "/",
                Mod => "%",
                Equal => "==",
                NotEqual => "!=",
                Less => "<",
                Greater => ">",
                LessEqual => "<=",
                GreaterEqual => ">=",
                And => "&&",
                Or => "||",
            }
            .to_string()
        }
    }

    impl ToString for Expr<String> {
        fn to_string(&self) -> String {
            use Expr::*;
            // we want to minmize the amount of parens in the expression
            match self {
                BinOp { lhs, op, rhs, .. } => {
                    let lhs_string = if lhs.precedence() <= self.precedence() {
                        format!("({})", lhs.to_string())
                    } else {
                        format!("{}", lhs.to_string())
                    };
                    let rhs_string = if rhs.precedence() <= self.precedence() {
                        format!("({})", rhs.to_string())
                    } else {
                        format!("{}", rhs.to_string())
                    };
                    format!("{} {} {}", lhs_string, op.to_string(), rhs_string)
                }
                Ter { cond, yes, no, .. } => format!(
                    "{} ? {} : {}",
                    cond.to_string(),
                    yes.to_string(),
                    no.to_string()
                ),
                Scalar(id) => id.to_string(),
                Nested(.., expr) => format!("({})", expr.to_string()),
                Neg(.., expr) => {
                    if expr.is_binop() {
                        format!("-({})", expr.to_string())
                    } else {
                        format!("- {}", expr.to_string())
                    }
                }
                Not(.., expr) => {
                    if expr.is_binop() {
                        format!("!({})", expr.to_string())
                    } else {
                        format!("!{}", expr.to_string())
                    }
                }
                Literal(lit) => lit.to_string(),
                e => todo!("{e:?}"),
            }
        }
    }
    proptest! {
        #[test]
        fn test_expr_to_string(arb_expr in expr()) {
            let expr_string = arb_expr.to_string();
            let mut parser = parser!(expr_string);
            let parsed = parser.expr().unwrap();
            prop_assert_eq!(arb_expr, parsed);
            prop_assert!(parser.finised());
        }
    }
}

mod trophies {
    use crate::{ast::*, span::*};

    #[test]
    fn literals() {
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
}

use crate::ast::{ELiteral, Expr};
use crate::lexer::Token::*;
use crate::parser::Error::*;

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
}
