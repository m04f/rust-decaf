use super::App;
use dcfrs::{lexer::*, semantics::*, span::*};

use std::fs::read_to_string;

pub struct Semantics;

impl App for Semantics {
    fn run(
        _stdout: &mut dyn std::io::Write,
        _stderr: &mut dyn std::io::Write,
        input_file: String,
    ) -> crate::ExitStatus {
        let text = read_to_string(input_file).unwrap();
        let code = SpanSource::new(text.as_bytes());
        let mut parser =
            dcfrs::parser::Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                eprintln!("{e:?}")
            });
        let proot = parser.doc_elems().collect();
        let hirtree = HIRRoot::from_proot(proot);
        match hirtree {
            Ok(_) => crate::ExitStatus::Success,
            Err(e) => {
                eprintln!("{e:?}");
                crate::ExitStatus::Fail
            }
        }
    }
}

#[cfg(test)]
mod test;
