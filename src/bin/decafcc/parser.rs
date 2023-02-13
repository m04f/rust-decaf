use std::fs::read_to_string;

use crate::*;
use dcfrs::lexer::*;

pub struct Parser;

impl App for Parser {
    fn run(
        _stdout: &mut dyn std::io::Write,
        _stderr: &mut dyn std::io::Write,
        input_file: String,
    ) -> ExitStatus {
        let text = read_to_string(input_file).unwrap();
        dcfrs::parser::Parser::new(
            tokens(text.as_bytes(), |_| panic!()).map(|s| s.map(|t| t.unwrap())),
            |_| panic!(),
        )
        .doc_elems()
        .for_each(|e| println!("{e:#?}"));
        ExitStatus::Success
    }
}
