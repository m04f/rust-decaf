use crate::error::CCError;
use crate::lexer::Token;
use crate::span::*;

use Error::*;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error<'a> {
    Expected {
        expected: Token,
        found: Token,
        span: Span<'a>,
    },
    ExpectedMatching {
        lspan: Span<'a>,
        left: Token,
        right: Token,
        rspan: Span<'a>,
    },
    ExpectedExpression(Span<'a>),
    ExpectedBlock(Span<'a>),
    ExpectedAssignExpr(Span<'a>),
    Unexpected(Token, Span<'a>),
    WrapInParens(Span<'a>),
    ImportAfterDecl {
        import_pos: Span<'a>,
        hinted_pos: Span<'a>,
    },
    ImportAfterFunc {
        import_pos: Span<'a>,
        hinted_pos: Span<'a>,
    },
    DeclAfterFunc {
        decl_pos: Span<'a>,
        hinted_pos: Span<'a>,
    },
    ForInitHasToBeAssign(Span<'a>),
    ForUpdateIsIncOrCompound(Span<'a>),
}

impl CCError for Error<'_> {
    fn msgs(&self) -> Vec<(String, (usize, usize))> {
        match self {
            Expected {
                expected,
                found,
                span,
            } => vec![(
                format!("expected token: {}, found: {}", expected, found),
                span.position(),
            )],
            ExpectedMatching {
                lspan,
                right,
                rspan,
                ..
            } => vec![(
                format!(
                    "expected matching: {} for the opening {} ",
                    right,
                    lspan.to_string()
                ),
                rspan.position(),
            )],
            ExpectedExpression(span) => vec![(
                format!("expected expression, found: {}", span.to_string()),
                span.position(),
            )],
            ExpectedBlock(span) => vec![(
                format!("expected block, found: {}", span.to_string()),
                span.position(),
            )],
            ExpectedAssignExpr(span) => vec![(
                format!("expected assign expression, found: {}", span.to_string()),
                span.position(),
            )],
            Unexpected(token, span) => {
                vec![(format!("unexpected token: {}", token), span.position())]
            }
            WrapInParens(span) => vec![(
                format!("wrap expression in parens: {}", span.to_string()),
                span.position(),
            )],
            ImportAfterDecl {
                import_pos,
                hinted_pos,
            } => vec![
                (
                    "imports have to be at the top of the file".to_string(),
                    import_pos.position(),
                ),
                (
                    "hint: move the import above the declaration".to_string(),
                    hinted_pos.position(),
                ),
            ],
            ImportAfterFunc {
                import_pos,
                hinted_pos,
            } => vec![
                (
                    "imports have to be at the top of the file".to_string(),
                    import_pos.position(),
                ),
                (
                    "hint: move the import to".to_string(),
                    hinted_pos.position(),
                ),
            ],
            DeclAfterFunc {
                decl_pos,
                hinted_pos,
            } => vec![
                (
                    "declarations have to be before function declaratiosn".to_string(),
                    decl_pos.position(),
                ),
                (
                    "hint: move the declaration to".to_string(),
                    hinted_pos.position(),
                ),
            ],
            ForInitHasToBeAssign(span) => vec![(
                "for init has to be an assign expression".to_string(),
                span.position(),
            )],
            ForUpdateIsIncOrCompound(span) => vec![(
                "for update has to be an increment or compound assign expression".to_string(),
                span.position(),
            )],
        }
    }
}
