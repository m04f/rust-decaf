use dcfrs::hir::*;
use dcfrs::lexer::*;
use dcfrs::parser::ast::PRoot;
use dcfrs::parser::*;
use dcfrs::span::*;
use std::io::Write;
use std::process::Command;
use std::process::Stdio;

macro_rules! test_valid {
    ($name:ident,$num:literal) => {
        #[test]
        fn $name() {
            let output_file = concat!("/tmp/decafrs-", stringify!($name));
            let code = SpanSource::new(include_bytes!(concat!(
                "../../decaf-tests/codegen/input/",
                stringify!($num),
                "-",
                stringify!($name),
                ".dcf"
            )));
            let expected = include_bytes!(concat!(
                "../../decaf-tests/codegen/output/",
                stringify!($num),
                "-",
                stringify!($name),
                ".dcf.out"
            ));
            let mut parser =
                Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                    unreachable!("{:?}", e)
                });
            let proot: PRoot = parser.doc_elems().collect();
            let program = HIRRoot::from_proot(proot).unwrap().destruct().codegen();
            let mut gcc_proc = Command::new("gcc")
                .args(["-x", "assembler", "-", "-o", output_file])
                .stdin(Stdio::piped())
                .spawn()
                .unwrap();
            gcc_proc
                .stdin
                .as_ref()
                .unwrap()
                .write_all(program.as_bytes())
                .unwrap();
            assert!(gcc_proc.wait().unwrap().success());
            let proc_output = Command::new(output_file).output().unwrap();
            // assert!(proc_output.status.success());
            assert_eq!(proc_output.stdout, expected);
        }
    };
    ($part1:ident-$part2:ident, $num:literal) => {
        #[test]
        fn $part1() {
            let output_file = concat!("/tmp/decafrs-", stringify!($part1_$part2));
            let code = SpanSource::new(include_bytes!(concat!(
                "../../decaf-tests/codegen/input/",
                stringify!($num),
                "-",
                stringify!($part1),
                "-",
                stringify!($part2),
                ".dcf"
            )));
            let expected = include_bytes!(concat!(
                "../../decaf-tests/codegen/output/",
                stringify!($num),
                "-",
                stringify!($part1),
                "-",
                stringify!($part2),
                ".dcf.out"
            ));
            let mut parser =
                Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                    unreachable!("{:?}", e)
                });
            let proot: PRoot = parser.doc_elems().collect();
            let program = HIRRoot::from_proot(proot).unwrap().destruct().codegen();
            let mut gcc_proc = Command::new("gcc")
                .args(["-x", "assembler", "-", "-o", output_file])
                .stdin(Stdio::piped())
                .spawn()
                .unwrap();
            gcc_proc
                .stdin
                .as_ref()
                .unwrap()
                .write_all(program.as_bytes())
                .unwrap();
            assert!(gcc_proc.wait().unwrap().success());
            let proc_output = Command::new(output_file).output().unwrap();
            // assert!(proc_output.status.success());
            assert_eq!(proc_output.stdout, expected);
        }
    };
    ($part1:ident-$part2:ident-$part3:ident, $num:literal) => {
        #[test]
        fn $part1() {
            let output_file = concat!("/tmp/decafrs-", stringify!($part1_$part2));
            let code = SpanSource::new(include_bytes!(concat!(
                "../../decaf-tests/codegen/input/",
                stringify!($num),
                "-",
                stringify!($part1),
                "-",
                stringify!($part2),
                "-",
                stringify!($part3),
                ".dcf"
            )));
            let expected = include_bytes!(concat!(
                "../../decaf-tests/codegen/output/",
                stringify!($num),
                "-",
                stringify!($part1),
                "-",
                stringify!($part2),
                "-",
                stringify!($part3),
                ".dcf.out"
            ));
            let mut parser =
                Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                    unreachable!("{:?}", e)
                });
            let proot: PRoot = parser.doc_elems().collect();
            let program = HIRRoot::from_proot(proot).unwrap().destruct().codegen();
            let mut gcc_proc = Command::new("gcc")
                .args(["-x", "assembler", "-", "-o", output_file])
                .stdin(Stdio::piped())
                .spawn()
                .unwrap();
            gcc_proc
                .stdin
                .as_ref()
                .unwrap()
                .write_all(program.as_bytes())
                .unwrap();
            assert!(gcc_proc.wait().unwrap().success());
            let proc_output = Command::new(output_file).output().unwrap();
            // assert!(proc_output.status.success());
            assert_eq!(proc_output.stdout, expected);
        }
    };
}

test_valid!(import, 01);
test_valid!(expr, 02);
test_valid!(math, 03);
test_valid!(math2, 04);
test_valid!(calls, 05);
test_valid!(control-flow, 06);
test_valid!(recursion, 07);
test_valid!(array, 08);
test_valid!(global, 09);
test_valid!(bounds, 10);
test_valid!(big-array, 11);
test_valid!(huge, 12);
test_valid!(ifs, 13);
test_valid!(shortcircuit, 14);
test_valid!(not, 15);
test_valid!(qsort, 16);
test_valid!(insertionsort, 17);
test_valid!(dead-code-bounds, 18);
