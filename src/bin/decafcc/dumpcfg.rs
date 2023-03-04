use super::App;
use dcfrs::{error::*, hir::*, lexer::*, span::*};

use std::fs::{self, read_to_string};

pub struct DumpCFG;

impl App for DumpCFG {
    fn run(
        stdout: &mut dyn std::io::Write,
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
            Ok(tree) => tree
                .functions
                .into_values()
                .map(|func| {
                    write!(
                        stdout,
                        "writing {name} cfg to {name}.dot\n",
                        name = func.name.to_string()
                    )
                    .and_then(|_| {
                        fs::write(
                            format!("{}.dot", func.name.to_string()),
                            func.destruct().to_dot(),
                        )
                    })
                })
                .fold(crate::ExitStatus::Success, |exit_stat, res| {
                    if res.is_ok() {
                        exit_stat
                    } else {
                        crate::ExitStatus::Fail
                    }
                }),
            Err(errs) => {
                errs.into_iter()
                    .try_for_each(|err| ewrite(stderr, &input_file, err))
                    .unwrap();
                crate::ExitStatus::Fail
            }
        }
    }
}
