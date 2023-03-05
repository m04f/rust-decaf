use super::App;
use dcfrs::{error::*, hir::*, lexer::*, span::*};

use std::{
    fs::{self, read_to_string},
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
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
                let out_dir = PathBuf::from("/tmp/decafcc/cfg-dump");
                fs::create_dir_all(&out_dir).unwrap();
                let output_files = tree
                    .functions
                    .into_values()
                    .map(|func| {
                        let out_file = out_dir.join(func.name.to_string()).with_extension("png");
                        let dot = func.destruct().to_dot();
                        let mut dot_proc = Command::new("dot")
                            .arg("-Tpng")
                            .arg("-o")
                            .arg(&out_file)
                            .stdin(Stdio::piped())
                            .spawn()
                            .unwrap();
                        dot_proc
                            .stdin
                            .as_ref()
                            .unwrap()
                            .write_all(dot.as_bytes())
                            .unwrap();
                        dot_proc.wait().unwrap();
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
