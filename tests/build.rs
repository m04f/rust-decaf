use dcfrs::hir::*;
use dcfrs::ir::Program;
use dcfrs::lexer::*;
use dcfrs::parser::ast::PRoot;
use dcfrs::parser::*;
use dcfrs::span::*;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

#[macro_use]
extern crate seq_macro;

macro_rules! gen_dcf_func {
    ($ret:ident, $name:ident, $argty:ident, $argc:literal, $sep:literal) => {
        seq!(N in 1..$argc {
            concat!(stringify!($ret), stringify!($name), "(", stringify!($argty), " i0" #(,stringify!(, $argty i~N))*, ")", "{",
              "return", "i0" #(,stringify!(, i~N))*, ";",
            "}")
        })
    };
}

seq!(N in 4..10 {
    const ADD: &[&str] = &[gen_dcf_func!(int, add, int, 3, "+") #(, gen_dcf_func!(int, add, int, N, "+"))*];
    const SUB: &[&str] = &[gen_dcf_func!(int, sub, int, 3, "-") #(,gen_dcf_func!(int, sub, int, N, "-"))*];
    const MUL: &[&str] = &[gen_dcf_func!(int, mul, int, 3, "-") #(, gen_dcf_func!(int, mul, int, N, "*"))*];
    const DIV: &[&str] = &[gen_dcf_func!(int, div, int, 3, "-") #(, gen_dcf_func!(int, div, int, N, "/"))*];
    const REM: &[&str] = &[gen_dcf_func!(int, rem, int, 3, "-") #(, gen_dcf_func!(int, rem, int, N, "%"))*];
    const LESS: &[&str] = &[gen_dcf_func!(int, less, int, 3, "-") #(, gen_dcf_func!(bool, less, int, N, "<"))*];
    const LESSEQ: &[&str] = &[gen_dcf_func!(int, lesseq, int, 3, "-") #(, gen_dcf_func!(bool, lesseq, int, N, "<="))*];
    const GREATER: &[&str] = &[gen_dcf_func!(int, greater, int, 3, "-") #(, gen_dcf_func!(bool, greater, int, N, ">"))*];
});

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/dcflib.dcf");
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = Path::new(&out_dir);
    let asm_path = &out_dir.join("dcflib.s");
    let obj_path = &out_dir.join("dcflib.o");
    let lib_path = &out_dir.join("libdcflib.a");

    let code = SpanSource::new(include_bytes!("src/dcflib.dcf"));
    let mut parser = Parser::new(tokens(code.source()).map(|s| s.map(|t| t.unwrap())), |e| {
        unreachable!("{:?}", e)
    });
    let proot: PRoot = parser.doc_elems().collect();
    let sym_table = HashMap::default();
    let vst = VSymMap::new(&sym_table);
    let func_table = HashMap::default();
    let fst = FSymMap::new(&func_table);
    let lib = Program::new(
        [],
        proot
            .funcs
            .into_iter()
            .map(|func| HIRFunction::from_pfunction(func, vst, fst).unwrap())
            .map(|hirfunc| hirfunc.destruct()),
    );
    let code = lib.codegen();
    fs::write(asm_path, code).unwrap();
    eprintln!("written asm to {}", asm_path.to_string_lossy());

    assert!(Command::new("gcc")
        .arg("-c",)
        .arg("-o",)
        .arg(obj_path)
        .arg(asm_path)
        .status()
        .unwrap()
        .success());

    assert!(Command::new("ar")
        .arg("crus")
        .arg(lib_path)
        .arg(obj_path)
        .status()
        .unwrap()
        .success());

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=dcflib");
}
