use std::fs::read_to_string;

use crate::*;
use dcfrs::lexer::*;

#[cfg(test)]
mod test;

pub struct Parser;

impl App for Parser {
    fn run(
        _stdout: &mut dyn std::io::Write,
        _stderr: &mut dyn std::io::Write,
        input_file: String,
    ) -> ExitStatus {
        let text = read_to_string(input_file).unwrap();
        let mut parser = dcfrs::parser::Parser::new(
            tokens(text.as_bytes(), |e| panic!("{e:?}")).map(|s| s.map(|t| t.unwrap())),
            |e| panic!("{e:?}"),
        );
        parser.doc_elems().for_each(|e| println!("{e:#?}"));
        if parser.finised() {
            ExitStatus::Success
        } else {
            ExitStatus::Fail
        }
    }
}
