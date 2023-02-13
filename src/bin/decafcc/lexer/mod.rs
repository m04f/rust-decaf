use dcfrs::lexer::{log_err, tokens};
use std::{fs, io::Read};

use crate::{App, ExitStatus};

pub struct Lexer;

impl App for Lexer {
    fn run(
        stdout: &mut dyn std::io::Write,
        stderr: &mut dyn std::io::Write,
        input_file: String,
    ) -> ExitStatus {
        /// shadows std's `println` macro
        macro_rules! println {
            ($($arg:tt)*) => ({
                writeln!(stdout, $($arg)*).unwrap();
            });
        }

        let ewrite = |s: String| writeln!(stderr, "{}", s).unwrap();
        let log_err = log_err(ewrite, input_file.clone());
        let mut buf = vec![];
        fs::File::open(input_file)
            .unwrap()
            .read_to_end(&mut buf)
            .unwrap();
        let err_count = tokens(&buf, log_err)
            .filter_map(|tok| {
                use dcfrs::lexer::Token::*;
                use std::string::String as StdString;
                let string = |slice: &[u8]| StdString::from_utf8(slice.to_vec()).unwrap();
                match tok.get() {
                    Ok(Eof) => None,
                    Ok(
                        Semicolon | And | Or | EqualEqual | NotEqual | Greater | GreaterEqual | Less
                        | LessEqual | Minus | Plus | Assign | SubAssign | AddAssign | Colon
                        | Question | Comma | Void | For | Continue | Break | While | Int | Bool
                        | If | Else | Return | Len | Star | Slash | Percent | Not | LeftParen
                        | RightParen | CurlyLeft | CurlyRight | SquareLeft | SquareRight
                        | Increment | Decrement | Import,
                    ) => {
                        println!("{} {}", tok.line(), string(tok.fragment()));
                        None
                    }
                    Ok(Identifier) => {
                        println!(
                            "{} IDENTIFIER {}",
                            tok.line(),
                            StdString::from_utf8(tok.fragment().to_vec()).unwrap()
                        );
                        None
                    }
                    Ok(DecimalLiteral | HexLiteral) => {
                        println!("{} INTLITERAL {}", tok.line(), string(tok.fragment()));
                        None
                    }
                    Ok(StringLiteral) => {
                        println!("{} STRINGLITERAL {}", tok.line(), string(tok.fragment()));
                        None
                    }
                    Ok(CharLiteral(_)) => {
                        println!("{} CHARLITERAL {}", tok.line(), string(tok.fragment()));
                        None
                    }
                    Ok(True | False) => {
                        println!("{} BOOLEANLITERAL {}", tok.line(), string(tok.fragment()));
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
}

#[cfg(test)]
mod test;
