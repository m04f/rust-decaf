use crate::ir::{Function, Instruction, Op, Reg};
use std::collections::HashMap;

/// decaf functions that are injected to the source code
///
/// TODO: improve the error messages

/// bound check function
const BOUND_CHECK: &str = r#"
void __bound_check(int index, int bound) {
    if (index < 0) {
        printf("array index can not be negative");
        exit(-1);
    } else if (index > bound) {
        printf("array index out of bound");
        exit(-1);
    } else {
        return;
    }
}
"#;

/// return guard function
const RETURN_GUARD: &str = r#"
void __return_guard() {
    printf("control reaced end of function without a return value");
    exit(-1);
}
"#;

/// enum to represent x86_64 registers.
#[derive(Debug, Clone, Copy)]
enum MacReg {
    Rax,
    Rcx,
    Rdx,
    Rbx,
    Rsi,
    Rdi,
    Rsp,
    Rbp,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

use MacReg::*;

const ARG_REGISTERS: [MacReg; 6] = [Rdi, Rsi, Rdx, Rcx, R8, R9];

impl MacReg {
    /// returns the x86_64 register name.
    fn name(&self) -> &'static str {
        match self {
            MacReg::Rax => "rax",
            MacReg::Rcx => "rcx",
            MacReg::Rdx => "rdx",
            MacReg::Rbx => "rbx",
            MacReg::Rsi => "rsi",
            MacReg::Rdi => "rdi",
            MacReg::Rsp => "rsp",
            MacReg::Rbp => "rbp",
            MacReg::R8 => "r8",
            MacReg::R9 => "r9",
            MacReg::R10 => "r10",
            MacReg::R11 => "r11",
            MacReg::R12 => "r12",
            MacReg::R13 => "r13",
            MacReg::R14 => "r14",
            MacReg::R15 => "r15",
        }
    }
}

impl Instruction {
    fn codegen(&self, vreg_stoffset: HashMap<Reg, usize>) -> Vec<String> {
        use Instruction::*;
        use crate::parser::ast::Op::*;
        match self {
            Op2 {
                dest,
                op,
                source1: lhs,
                source2: rhs,
            } => match op {
                Add  => {
                    let mut code = Vec::new();
                    let lhs = vreg_stoffset.get(lhs).unwrap();
                    let rhs = vreg_stoffset.get(rhs).unwrap();
                    let dest = vreg_stoffset.get(dest).unwrap();
                    code.push(format!("mov rax, [rbp - {}]", lhs));
                    code.push(format!("add rax, [rbp - {}]", rhs));
                    code.push(format!("mov [rbp - {}], rax", dest));
                    code
                }
                Sub  => {
                    let mut code = Vec::new();
                    let lhs = vreg_stoffset.get(lhs).unwrap();
                    let rhs = vreg_stoffset.get(rhs).unwrap();
                    let dest = vreg_stoffset.get(dest).unwrap();
                    code.push(format!("mov rax, [rbp - {}]", lhs));
                    code.push(format!("sub rax, [rbp - {}]", rhs));
                    code.push(format!("mov [rbp - {}], rax", dest));
                    code
                }
                Mul { dest, lhs, rhs } => {
                    let mut code = Vec::new();
                    let lhs = vreg_stoffset.get(lhs).unwrap();
                    let rhs = vreg_stoffset.get(rhs).unwrap();
                    let dest = vreg_stoffset.get(dest).unwrap();
                    code.push(format!("mov rax, [rbp - {}]", lhs));
                    code.push(format!("imul rax, [rbp - {}]", rhs));
                    code.push(format!("mov [rbp - {}], rax", dest));
                    code
                }
                Div { dest, lhs, rhs } => {
                    let mut code = Vec::new();
                    let lhs = vreg_stoffset.get(lhs).unwrap();
                    let rhs = vreg_stoffset.get(rhs).unwrap();
                    let dest = vreg_stoffset.get(dest).unwrap();
                    code.push(format!("mov rax, [rbp - {}]", lhs));
                    code.push(format!("cqo"));
                    code.push(format!("idiv [rbp - {}]", rhs));
                    code.push(format!("mov [rbp - {}], rax", dest));
                    code
                }
            },
        }
    }
}

impl Function {
    /// generates the x86_64 code for the function.
    pub fn codegen(&self) -> String {
        todo!()
    }
}
