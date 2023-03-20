use crate::hir::*;
use crate::lexer::*;
use crate::parser::ast::PRoot;
use crate::parser::*;
use crate::span::*;

macro_rules! test_valid {
    ($name:ident,$num:literal) => {
        #[test]
        fn $name() {
            let code = SpanSource::new(include_bytes!(concat!(
                "../../decaf-tests/codegen/input/",
                stringify!($num),
                "-",
                stringify!($name),
                ".dcf"
            )));
            let mut parser =
                Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                    unreachable!("{:?}", e)
                });
            let proot: PRoot = parser.doc_elems().collect();
            let _program = HIRRoot::from_proot(proot, true)
                .unwrap()
                .destruct()
                .codegen();
        }
    };
    ($part1:ident-$part2:ident, $num:literal) => {
        #[test]
        fn $part1() {
            let code = SpanSource::new(include_bytes!(concat!(
                "../../decaf-tests/codegen/input/",
                stringify!($num),
                "-",
                stringify!($part1),
                "-",
                stringify!($part2),
                ".dcf"
            )));
            let mut parser =
                Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                    unreachable!("{:?}", e)
                });
            let proot: PRoot = parser.doc_elems().collect();
            let _program = HIRRoot::from_proot(proot, true)
                .unwrap()
                .destruct()
                .codegen();
        }
    };
    ($part1:ident-$part2:ident-$part3:ident, $num:literal) => {
        #[test]
        fn $part1() {
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
            let mut parser =
                Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
                    unreachable!("{:?}", e)
                });
            let proot: PRoot = parser.doc_elems().collect();
            let _program = HIRRoot::from_proot(proot, true)
                .unwrap()
                .destruct()
                .codegen();
        }
    };
}

test_valid!(import, 01);
test_valid!(expr, 02);
test_valid!(math, 03);
test_valid!(math2, 04);
test_valid!(calls, 05);
test_valid!(control - flow, 06);
test_valid!(recursion, 07);
test_valid!(array, 08);
test_valid!(global, 09);
test_valid!(bounds, 10);
test_valid!(big - array, 11);
test_valid!(huge, 12);
test_valid!(ifs, 13);
test_valid!(shortcircuit, 14);
test_valid!(not, 15);
test_valid!(qsort, 16);
test_valid!(insertionsort, 17);
test_valid!(dead - code - bounds, 18);
