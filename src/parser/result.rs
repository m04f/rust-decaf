use std::fmt::Debug;

use self::Result::*;
use crate::lexer::Token;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Expected(Token, Token),
    ExpectedExpression,
    ExpectedBlock,
    LenNoArg,
    Syn(Token),
    ExpectedMatching(Token, Token),
    ExpectedAssignExpr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Result<T> {
    Parsed(T),
    ErrorWithResult(T),
    Error,
    Nil,
}

impl<T> Result<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Result<U> {
        match self {
            Parsed(t) => Parsed(f(t)),
            ErrorWithResult(t) => ErrorWithResult(f(t)),
            Error => Error,
            Nil => Nil,
        }
    }

    pub fn is_nil(&self) -> bool {
        match self {
            Nil => true,
            _ => false,
        }
    }

    pub fn is_error(&self) -> bool {
        match self {
            Error => true,
            _ => false,
        }
    }

    pub fn has_output(&self) -> bool {
        match self {
            Parsed(_) | ErrorWithResult(_) => true,
            _ => false,
        }
    }

    // TODO: rename this
    pub fn has_error(&self) -> bool {
        match self {
            Error | ErrorWithResult(_) => true,
            _ => false,
        }
    }

    pub fn or_else(self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        match self {
            Nil => f(),
            _ => self,
        }
    }

    pub fn into_parsed_error(self) -> Self {
        match self {
            Parsed(t) => ErrorWithResult(t),
            _ => self,
        }
    }

    pub fn unwrap_parsed(self) -> T where T: Debug {
        match self {
            Parsed(t) => t,
            _ => panic!("Expected parsed result found: {self:#?}"),
        }
    }

    pub fn and<U>(self, other: Result<U>) -> Result<(T, U)> {
        match (self, other) {
            (Parsed(t), Parsed(u)) => Parsed((t, u)),
            (ErrorWithResult(t), Parsed(u)) => ErrorWithResult((t, u)),
            (Parsed(t), ErrorWithResult(u)) => ErrorWithResult((t, u)),
            (ErrorWithResult(t), ErrorWithResult(u)) => ErrorWithResult((t, u)),
            (Nil, Nil) => Nil,
            _ => Error,
        }
    }

    pub fn unwrap_output(self) -> T {
        match self {
            Parsed(t) => t,
            ErrorWithResult(t) => t,
            _ => panic!("expected output"),
        }
    }

    pub fn and_then<U>(self, f: impl FnOnce(T) -> Result<U>) -> Result<U> {
        match self {
            Parsed(t) => f(t),
            ErrorWithResult(t) => f(t).into_parsed_error(),
            Nil => Nil,
            Error => Error,
        }
    }

    pub fn map_nil(self, f: impl FnOnce() -> Self) -> Self {
        match self {
            Nil => f(),
            _ => self,
        }
    }
}
