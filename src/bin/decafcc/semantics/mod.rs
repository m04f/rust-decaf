use super::App;
use dcfrs::{error::*, lexer::*, hir::*, span::*};

use std::fs::read_to_string;

pub struct Semantics;

impl App for Semantics {
    fn run(
        _stdout: &mut dyn std::io::Write,
        stderr: &mut dyn std::io::Write,
        input_file: String,
    ) -> crate::ExitStatus {
        let text = read_to_string(&input_file).unwrap();
        let code = SpanSource::new(text.as_bytes());
        let mut parser =
            dcfrs::parser::Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                ewrite(stderr, &input_file, e).unwrap();
            });
        let proot = parser.doc_elems().collect();
        let hirtree = HIRRoot::from_proot(proot);
        match hirtree {
            Ok(_) => crate::ExitStatus::Success,
            Err(errs) => {
                errs.into_iter()
                    .try_for_each(|err| ewrite(stderr, &input_file, err))
                    .unwrap();
                crate::ExitStatus::Fail
            }
        }
    }
}

#[cfg(test)]
mod test;
