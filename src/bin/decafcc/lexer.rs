use dcfrs::lexer;

use crate::ExitStatus;

pub fn run(code: &[u8]) -> ExitStatus {
    let err_count = lexer::tokens(code)
        .filter_map(|tok| {
            use dcfrs::lexer::Token::*;
            use std::string::String as StdString;
            let string = |slice: &[u8]| StdString::from_utf8(slice.to_vec()).unwrap();
            match tok.get() {
                Ok(Eof) => None,
                Ok(
                    Semicolon | And | Or | Equal | NotEqual | Greater | GreaterEqual | Less
                    | LessEqual | Minus | Plus | Assign | SubAssign | AddAssign | Colon | Question
                    | Comma | Void | For | Continue | Break | While | Int | Bool | If | Else
                    | Return | False | True | Len | Star | Slash | Percent | Not | LeftParen
                    | RightParen | CurlyLeft | CurlyRight | SquareLeft | SquareRight,
                ) => {
                    println!("{}: {}", tok.line(), string(tok.fragment()));
                    None
                }
                Ok(Identifier) => {
                    println!(
                        "{}: IDENTIFIER {}",
                        tok.line(),
                        StdString::from_utf8(tok.fragment().to_vec()).unwrap()
                    );
                    None
                }
                Ok(DecimalLiteral | HexLiteral) => {
                    println!("{}: INTLITERAL {}", tok.line(), string(tok.fragment()));
                    None
                }
                Ok(String) => {
                    println!("{}: STRINGLITERAL {}", tok.line(), string(tok.fragment()));
                    None
                }
                Ok(Char(_)) => {
                    println!("{}: CHARLITERAL {}", tok.line(), string(tok.fragment()));
                    None
                }
                // errors are logged in the lexer module anyways
                Err(_) => Some(()),
                _ => unreachable!(),
            }
        })
        .count();
    if err_count == 0 {
        ExitStatus::Success
    } else {
        ExitStatus::Fail
    }
}
