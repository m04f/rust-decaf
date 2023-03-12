use super::App;
use dcfrs::{error::*, hir::*, ir::Function, lexer::*, span::*};

use std::{
    fs::{self, read_to_string},
    path::PathBuf,
    process::Command,
};

pub struct DumpCFG;

impl App for DumpCFG {
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
            Ok(tree) => {
                let out_dir = PathBuf::from("/tmp/decafcc/cfg-dump").join(input_file);
                fs::create_dir_all(&out_dir).unwrap();
                let out_file =
                    |function: &Function| out_dir.join(function.name()).with_extension("svg");
                let output_files = tree.destruct()
                    .functions()
                    .iter()
                    .map(|function| {
                        let out_file = out_file(function);
                        function.to_dot().compile(&out_file).unwrap();
                        out_file
                    })
                    .collect::<Vec<_>>();
                Command::new("imv").args(output_files).output().unwrap();
                crate::ExitStatus::Success
            }
            Err(errs) => {
                errs.into_iter()
                    .try_for_each(|err| ewrite(stderr, &input_file, err))
                    .unwrap();
                crate::ExitStatus::Fail
            }
        }
    }
}
