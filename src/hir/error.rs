use crate::{error::*, parser::ast::Type, span::*};

#[derive(Debug)]
pub enum Error<'a> {
    UndeclaredIdentifier(Span<'a>),
    ExpectedArrayVariable(Span<'a>),
    ExpectedScalarVariable(Span<'a>),
    CannotIndexScalar(Span<'a>),
    CannotAssignToArray(Span<'a>),
    ExpectedBoolExpr(Span<'a>),
    ExpectedIntExpr(Span<'a>),
    ReturnValueFromVoid(Span<'a>),
    Redifinition(Span<'a>, Span<'a>),
    BreakOutsideLoop(Span<'a>),
    ContinueOutsideLoop(Span<'a>),
    VoidFuncAsExpr(Span<'a>),
    TypeMismatch {
        lhs: Type,
        rhs: Type,
        lspan: Span<'a>,
        rspan: Span<'a>,
    },
    WrongNumberOfArgs {
        expected: usize,
        found: usize,
        span: Span<'a>,
    },
    ExpectedType {
        expected: Type,
        found: Type,
        span: Span<'a>,
    },
    ExpectedExpression(Span<'a>),
    ZeroArraySize(Span<'a>),
    TooLargeInt(Span<'a>),
    RootDoesNotContainMain,
    InvalidMainSig(Span<'a>),
    VariableNotAMethod(Span<'a>),
    StringInUserDefined(Span<'a>),
    AssignOfDifferentType {
        lhs: Span<'a>,
        ltype: Type,
        rtype: Type,
    },
    IncNonInt(Span<'a>),
    DecNonInt(Span<'a>),
}

impl CCError for Error<'_> {
    fn msgs(self) -> Vec<(String, (usize, usize))> {
        match self {
            Self::AssignOfDifferentType { lhs, ltype, rtype } => {
                vec![(
                    format!(
                        "cannot assign value of type `{}` to variable `{}` of type `{}`",
                        rtype,
                        lhs.to_string(),
                        ltype
                    ),
                    lhs.position(),
                )]
            }
            Self::IncNonInt(span) => {
                vec![(
                    format!(
                        "cannot increment non-integer variable: {}",
                        span.to_string()
                    ),
                    span.position(),
                )]
            }
            Self::DecNonInt(span) => {
                vec![(
                    format!(
                        "cannot decrement non-integer variable: {}",
                        span.to_string()
                    ),
                    span.position(),
                )]
            }
            Self::UndeclaredIdentifier(span) => vec![(
                format!("Undeclared identifier `{}`", span.to_string()),
                span.position(),
            )],
            Self::ExpectedArrayVariable(span) => vec![(
                format!("Expected array variable, found `{}`", span.to_string()),
                span.position(),
            )],
            Self::ExpectedScalarVariable(span) => vec![(
                format!("Expected scalar variable, found `{}`", span.to_string()),
                span.position(),
            )],
            Self::CannotIndexScalar(span) => vec![(
                format!("Cannot index scalar variable `{}`", span.to_string()),
                span.position(),
            )],
            Self::CannotAssignToArray(span) => vec![(
                format!("Cannot assign to array variable `{}`", span.to_string()),
                span.position(),
            )],
            Self::ExpectedBoolExpr(span) => vec![(
                format!("Expected boolean expression, found `{}`", span.to_string()),
                span.position(),
            )],
            Self::ExpectedIntExpr(span) => vec![(
                format!("Expected integer expression, found `{}`", span.to_string()),
                span.position(),
            )],
            Self::ReturnValueFromVoid(span) => vec![(
                format!(
                    "Cannot return value from void function `{}`",
                    span.to_string()
                ),
                span.position(),
            )],
            Self::Redifinition(lhs, rhs) => vec![
                (
                    format!("Redifinition of `{}`", lhs.to_string()),
                    lhs.position(),
                ),
                (
                    format!("Previous definition of `{}`", rhs.to_string()),
                    rhs.position(),
                ),
            ],
            Self::BreakOutsideLoop(span) => vec![(
                format!("Break outside loop `{}`", span.to_string()),
                span.position(),
            )],
            Self::ContinueOutsideLoop(span) => vec![(
                format!("Continue outside loop `{}`", span.to_string()),
                span.position(),
            )],
            Self::VoidFuncAsExpr(span) => vec![(
                format!("void function `{}` used as expression", span.to_string()),
                span.position(),
            )],
            Self::TypeMismatch {
                lhs,
                rhs,
                lspan,
                rspan,
            } => vec![(
                format!(
                    "type mismatch:\n lhs: `{}` has type {:?}\n rhs: `{}` has type {:?}",
                    lspan.to_string(),
                    lhs,
                    rspan.to_string(),
                    rhs
                ),
                lspan.position(),
            )],
            Self::WrongNumberOfArgs {
                expected,
                found,
                span,
            } => vec![(
                format!(
                    "wrong number of arguments to function `{}`:\n expected: {}\n found: {}",
                    span.to_string(),
                    expected,
                    found
                ),
                span.position(),
            )],
            Self::ExpectedType {
                expected,
                found,
                span,
            } => vec![(
                format!(
                    "expected type `{}` for `{}`, but it hast type `{}`",
                    expected,
                    span.to_string(),
                    found
                ),
                span.position(),
            )],
            Self::ExpectedExpression(span) => vec![(
                format!("expected expression, found `{}`", span.to_string()),
                span.position(),
            )],
            Self::ZeroArraySize(span) => vec![(
                format!("array size cannot be zero `{}`", span.to_string()),
                span.position(),
            )],
            Self::TooLargeInt(span) => vec![(
                format!("integer literal is too large `{}`", span.to_string()),
                span.position(),
            )],
            Self::RootDoesNotContainMain => {
                vec![("root does not contain main function".to_string(), (0, 0))]
            }
            Self::InvalidMainSig(span) => vec![
                (
                    format!("main function has invalid signature `{}`", span.to_string()),
                    span.position(),
                ),
                (
                    "correct main signature is `void main()`".to_string(),
                    span.position(),
                ),
            ],
            Self::VariableNotAMethod(span) => vec![(
                format!("variable `{}` is not a method", span.to_string()),
                span.position(),
            )],
            Self::StringInUserDefined(span) => vec![(
                format!("string literal `{}` in user defined type", span.to_string()),
                span.position(),
            )],
        }
    }
}
