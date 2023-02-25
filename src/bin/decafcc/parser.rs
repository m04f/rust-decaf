use std::fs::read_to_string;

use crate::*;
use dcfrs::{error::ewrite, lexer::*, span::SpanSource};

#[cfg(test)]
mod test;

pub struct Parser;

impl App for Parser {
    fn run(
        _stdout: &mut dyn std::io::Write,
        stderr: &mut dyn std::io::Write,
        input_file: String,
    ) -> ExitStatus {
        let text = read_to_string(&input_file).unwrap();
        let code = SpanSource::new(text.as_bytes());
        let mut parser =
            dcfrs::parser::Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                ewrite(stderr, &input_file, e).unwrap()
            });
        parser.doc_elems().for_each(|_| {});
        if parser.finised() && !parser.found_errors() {
            ExitStatus::Success
        } else {
            ExitStatus::Fail
        }
    }
}
