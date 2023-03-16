use super::App;
use dcfrs::hir::*;
use dcfrs::lexer::*;
use dcfrs::parser::*;
use dcfrs::span::*;
use std::fs::read_to_string;

pub struct Asm;

impl App for Asm {
    fn run(
        stdout: &mut dyn std::io::Write,
        _stderr: &mut dyn std::io::Write,
        input_file: String,
    ) -> crate::ExitStatus {
        let text = read_to_string(input_file).unwrap();
        let code = SpanSource::new(text.as_bytes());
        let mut parser = Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
            unreachable!("{:?}", e)
        });
        let proot = parser.doc_elems().collect();
        let hirtree = HIRRoot::from_proot(proot).unwrap();
        write!(stdout, "{}", hirtree.destruct().codegen()).unwrap();
        crate::ExitStatus::Success
    }
}
