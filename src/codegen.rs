use crate::ir::{
    Dest, Function, IRExternArg, Immediate, Instruction, NeighbouringNodes, Op, Program, Source,
    Symbol, Unary,
};
use std::{collections::HashMap, fmt::Display};

use MacInstruction::*;
use MacReg::*;

/// enum to represent x86_64 registers.
#[allow(unused)]
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
    Rip,
    R8,
    R9,
    R10,
    R11,
    R12,
    R13,
    R14,
    R15,
}

#[derive(Debug, Clone, Copy)]
#[allow(unused)]
enum MacInstruction {
    Lea,
    Mov,
    Call,
    Jmp,
    Add,
    Sub,
    Mul,
}

impl Display for MacInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Lea => write!(f, "leaq"),
            Mov => write!(f, "movq"),
            Call => write!(f, "call"),
            Jmp => write!(f, "jmp"),
            Add => write!(f, "addq"),
            Sub => write!(f, "subq"),
            Mul => write!(f, "imulq"),
        }
    }
}

#[allow(unused)]
#[derive(Debug, Clone, Copy)]
enum MacOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Less,
    LessEq,
    Greater,
    GreaterEq,
    Equal,
    NotEqual,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Cmp,
}

impl MacOp {
    fn is_bool(&self) -> bool {
        matches!(
            self,
            MacOp::Less
                | MacOp::LessEq
                | MacOp::Greater
                | MacOp::GreaterEq
                | MacOp::Equal
                | MacOp::NotEqual
        )
    }
}

impl From<Op> for MacOp {
    fn from(value: Op) -> Self {
        match value {
            Op::Add => MacOp::Add,
            Op::Sub => MacOp::Sub,
            Op::Mul => MacOp::Mul,
            Op::Div => MacOp::Div,
            Op::Mod => MacOp::Mod,
            Op::Less => MacOp::Less,
            Op::Greater => MacOp::Greater,
            Op::LessEqual => MacOp::LessEq,
            Op::GreaterEqual => MacOp::GreaterEq,
            Op::Equal => MacOp::Equal,
            Op::NotEqual => MacOp::NotEqual,
            op => unimplemented!("{op:?}"),
        }
    }
}

const ARG_REGISTERS: [MacReg; 6] = [Rdi, Rsi, Rdx, Rcx, R8, R9];

impl MacReg {
    /// returns the x86_64 register name.
    fn name(&self) -> &'static str {
        match self {
            MacReg::Rax => "%rax",
            MacReg::Rcx => "%rcx",
            MacReg::Rdx => "%rdx",
            MacReg::Rbx => "%rbx",
            MacReg::Rsi => "%rsi",
            MacReg::Rdi => "%rdi",
            MacReg::Rsp => "%rsp",
            MacReg::Rbp => "%rbp",
            MacReg::Rip => "%rip",
            MacReg::R8 => "%r8",
            MacReg::R9 => "%r9",
            MacReg::R10 => "%r10",
            MacReg::R11 => "%r11",
            MacReg::R12 => "%r12",
            MacReg::R13 => "%r13",
            MacReg::R14 => "%r14",
            MacReg::R15 => "%r15",
        }
    }
}

impl Display for MacReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Display for MacOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use MacOp::*;
        match self {
            Add => write!(f, "addq"),
            Sub => write!(f, "subq"),
            // HACK: this is used for shorter codegen code.
            Less => write!(f, "cmovl"),
            LessEq => write!(f, "cmovle"),
            Greater => write!(f, "cmovg"),
            GreaterEq => write!(f, "cmovge"),
            Equal => write!(f, "cmovz"),
            NotEqual => write!(f, "cmovnz"),
            _ => todo!(),
        }
    }
}

#[derive(Clone)]
enum ExternArg {
    Source(MacVar),
    String(String, usize),
}

impl ExternArg {
    fn string_symbol(&self) -> Option<String> {
        match self {
            Self::Source(_) => None,
            Self::String(_, i) => Some(format!(".Cstr{i}")),
        }
    }
    fn string(&self) -> Option<&String> {
        match self {
            Self::Source(_) => None,
            Self::String(s, ..) => Some(s),
        }
    }
}

impl Display for ExternArg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(s) => write!(f, "{s}"),
            Self::String(..) => write!(f, "{}(%rip)", self.string_symbol().unwrap()),
        }
    }
}

enum LIr<'a> {
    BinOp {
        dest: MacVar,
        lhs: MacVar,
        rhs: MacVar,
        op: MacOp,
    },
    Unary {
        dest: MacVar,
        source: MacVar,
        op: Unary,
    },
    Mov {
        dest: MacVar,
        source: MacVar,
    },
    Select {
        dest: MacVar,
        cond: MacVar,
        yes: MacVar,
        no: MacVar,
    },
    ReturnGuard,
    VoidCall {
        symbol: &'a str,
        args: Vec<MacVar>,
    },
    Call {
        dest: MacVar,
        symbol: &'a str,
        args: Vec<MacVar>,
    },
    ExternCall {
        dest: MacVar,
        symbol: &'a str,
        args: Vec<ExternArg>,
    },
    Return {
        value: MacVar,
    },
    VoidReturn,
}

impl<'a> LIr<'a> {
    fn from_ir(
        instruction: &Instruction<'a>,
        sym_table: impl Fn(Symbol<'a>) -> Option<i64>,
        reg_table: impl Fn(u32) -> i64,
        mut bound_table: impl FnMut(Symbol<'a>) -> u32,
        const_strings: &mut usize,
    ) -> Option<(Self, Option<String>)> {
        match instruction {
            Instruction::Select {
                dest,
                cond,
                yes,
                no,
            } => Some((
                Self::Select {
                    dest: MacVar::from_dest(dest, &sym_table, &reg_table, &mut bound_table),
                    cond: MacVar::from_source(cond, &sym_table, &reg_table, &mut bound_table),
                    yes: MacVar::from_source(yes, &sym_table, &reg_table, &mut bound_table),
                    no: MacVar::from_source(no, &sym_table, &reg_table, &mut bound_table),
                },
                None,
            )),
            Instruction::Move { dest, source } => Some((
                Self::Mov {
                    dest: MacVar::from_dest(dest, &sym_table, &reg_table, &mut bound_table),
                    source: MacVar::from_source(source, &sym_table, &reg_table, &mut bound_table),
                },
                None,
            )),
            Instruction::Op2 {
                dest,
                lhs: source1,
                rhs: source2,
                op,
            } => Some((
                Self::BinOp {
                    dest: MacVar::from_dest(dest, &sym_table, &reg_table, &mut bound_table),
                    lhs: MacVar::from_source(source1, &sym_table, &reg_table, &mut bound_table),
                    rhs: MacVar::from_source(source2, &sym_table, &reg_table, &mut bound_table),
                    op: (*op).into(),
                },
                None,
            )),
            Instruction::Unary { dest, source, op } => Some((
                Self::Unary {
                    dest: MacVar::from_dest(dest, &sym_table, &reg_table, &mut bound_table),
                    source: MacVar::from_source(source, &sym_table, &reg_table, &mut bound_table),
                    op: *op,
                },
                None,
            )),
            Instruction::Return { value } => Some((
                Self::Return {
                    value: MacVar::from_source(value, sym_table, reg_table, &mut bound_table),
                },
                None,
            )),

            Instruction::ExternCall { dest, symbol, args } => Some({
                let args = args
                    .iter()
                    .map(|arg| match arg {
                        IRExternArg::Source(source) => ExternArg::Source(MacVar::from_source(
                            source,
                            &sym_table,
                            &reg_table,
                            &mut bound_table,
                        )),
                        IRExternArg::String(string) => ExternArg::String(string.to_string(), {
                            *const_strings += 1;
                            *const_strings
                        }),
                    })
                    .collect::<Vec<_>>();
                let const_defs = args
                    .iter()
                    .filter_map(|arg| arg.string_symbol().map(|sym| (sym, arg.string().unwrap())))
                    .map(|(sym, string)| {
                        format!(
                            r#"
{sym}:
    .string {string}
"#,
                            sym = sym
                        )
                    })
                    .collect();
                (
                    Self::ExternCall {
                        dest: MacVar::from_dest(dest, &sym_table, &reg_table, &mut bound_table),
                        symbol,
                        args,
                    },
                    Some(const_defs),
                )
            }),

            Instruction::ReturnGuard => Some((Self::ReturnGuard, None)),
            Instruction::VoidReturn => Some((Self::VoidReturn, None)),
            Instruction::VoidCall { symbol, args } => Some((
                Self::VoidCall {
                    symbol,
                    args: args
                        .iter()
                        .map(|source| {
                            MacVar::from_source(
                                &Source::from(*source),
                                &sym_table,
                                &reg_table,
                                &mut bound_table,
                            )
                        })
                        .collect(),
                },
                None,
            )),
            Instruction::AllocScalar { .. } | Instruction::AllocArray { .. } => None,
            Instruction::Call { dest, symbol, args } => Some((
                Self::Call {
                    dest: MacVar::from_dest(dest, &sym_table, &reg_table, &mut bound_table),
                    symbol,
                    args: args
                        .iter()
                        .map(|source| {
                            MacVar::from_source(
                                &Source::from(*source),
                                &sym_table,
                                &reg_table,
                                &mut bound_table,
                            )
                        })
                        .collect(),
                },
                None,
            )),
        }
    }
    fn codegen<S1: Display, S2: Display>(
        &self,
        on_bound_fail: impl FnOnce() -> S1,
        on_no_return: impl FnOnce() -> S2,
    ) -> String {
        match self {
            Self::Unary {
                dest,
                source,
                op: Unary::Not,
            } => {
                format!(
                    r#"
movq {source}, {R10}
movq $0, {R11}
cmp {R10}, {R11}
movq $1, {R10}
movq $0, {R11}
cmove {R10}, {R11}
movq {R11}, {dest}"#
                )
            }
            Self::Unary {
                dest,
                source,
                op: Unary::Neg,
            } => {
                format!(
                    r#"
movq {source}, {R10}
negq {R10}
movq {R10}, {dest}"#
                )
            }
            Self::Mov {
                dest: MacVar::Index { .. },
                source: MacVar::Index { .. },
            } => unimplemented!(),
            Self::Mov {
                dest: MacVar::Index { base, index },
                source,
            } => {
                let bound = base.size();
                format!(
                    r#"
movq $0, {R10}
movq {index}, {R11}
cmp {R11}, {R10}
jg {bound_fail}
movq ${bound}, {R10}
cmp {R11}, {R10}
jle {bound_fail}
leaq {base}, {Rax}
movq {index}, {R10}
movq {source}, {R11}
movq {R11}, ({Rax}, {R10}, 8)"#,
                    bound_fail = on_bound_fail()
                )
            }
            Self::Mov {
                dest,
                source: MacVar::Index { base, index },
            } => {
                let bound = base.size();
                format!(
                    r#"
movq $0, {R10}
movq {index}, {R11}
cmp {R11}, {R10}
jg {bound_fail}
movq ${bound}, {R10}
cmp {R11}, {R10}
jle {bound_fail}
leaq {base}, {Rax}
movq {index}, {R10}
movq ({Rax}, {R10}, 8), {R11}
movq {R11}, {dest}"#,
                    bound_fail = on_bound_fail()
                )
            }
            Self::Mov { dest, source } => {
                format!(
                    r#"
movq {source}, {R10}
movq {R10}, {dest}"#
                )
            }
            Self::BinOp {
                dest,
                lhs: source1,
                rhs: source2,
                op: MacOp::Mod,
            } => {
                format!(
                    r#"
movq {source1}, {Rax}
cqto
idivq {source2}
movq {Rdx}, {dest}"#
                )
            }
            Self::BinOp {
                dest,
                lhs: source1,
                rhs: source2,
                op: MacOp::Div,
            } => {
                format!(
                    r#"
movq {source1}, {Rax}
cqto
idivq {source2}
movq {Rax}, {dest}"#
                )
            }
            Self::BinOp { dest, lhs, rhs, op } if op.is_bool() => {
                format!(
                    r#"
movq {lhs}, {R10}
movq {rhs}, {R11}
cmp {R11}, {R10}
movq $1, {R10}
movq $0, {R11}
{op} {R10}, {R11}
movq {R11}, {dest}"#
                )
            }
            Self::BinOp {
                dest,
                lhs: source1,
                rhs: source2,
                op: MacOp::Mul,
            } => {
                format!(
                    r#"
movq {source1}, {Rax}
imulq {source2}
movq {Rax}, {dest}"#
                )
            }
            Self::BinOp { dest, op, lhs, rhs } => {
                let op = *op;
                format!(
                    r#"
movq {rhs}, {R10}
movq {lhs}, {R11}
{op} {R10}, {R11}
movq {R11}, {dest}"#
                )
            }
            Self::VoidReturn => r#"
leave
ret"#
                .to_string(),
            Self::Return { value } => {
                format!(
                    r#"
movq {value}, {Rax}
leave
ret"#
                )
            }
            Self::Select {
                dest,
                cond,
                yes,
                no,
            } => {
                format!(
                    r#"
movq {cond}, {R10}
movq $1, {R11}
cmp {R10}, {R11}
movq {yes}, {R10}
movq {no}, {R11}
cmove {R10}, {R11}
movq {R11}, {dest}
"#
                )
            }
            LIr::ReturnGuard => format!(
                r#"
{Jmp} {no_return}
"#,
                no_return = on_no_return()
            ),
            LIr::ExternCall { dest, symbol, args } => {
                let move_args = args
                    .iter()
                    .skip(ARG_REGISTERS.len())
                    .rev()
                    .map(|arg| match arg {
                        ExternArg::Source(s) => format!("\npushq {s}"),
                        ExternArg::String(..) => format!(
                            r#"
leaq {arg}, %rax
pushq {Rax}"#,
                        ),
                    })
                    .chain(
                        args.iter()
                            .zip(ARG_REGISTERS.iter())
                            .map(|(arg, reg)| match arg {
                                ExternArg::Source(s) => format!("\nmovq {s}, {reg}"),
                                ExternArg::String(..) => format!(
                                    r#"
leaq {arg}, %rax
movq {Rax}, {reg}"#,
                                ),
                            }),
                    )
                    .collect::<String>();
                format!(
                    r#"
{move_args}
movq $0, {Rax}
call {symbol}@PLT
movq {Rax}, {dest}"#
                )
            }
            LIr::VoidCall { symbol, args } => {
                let move_args = args
                    .iter()
                    .skip(ARG_REGISTERS.len())
                    .rev()
                    .map(|arg| format!("\npushq {arg}"))
                    .chain(
                        args.iter()
                            .zip(ARG_REGISTERS.iter())
                            .map(|(arg, reg)| format!("\nmovq {arg}, {reg}")),
                    )
                    .collect::<String>();
                format!(
                    r#"
subq ${args_frame}, {Rsp}
{move_args}
call {symbol}
addq ${args_frame}, {Rsp}"#,
                    args_frame = 8 * (args.len().saturating_sub(6))
                )
            }
            LIr::Call { dest, symbol, args } => {
                let move_args = args
                    .iter()
                    .skip(ARG_REGISTERS.len())
                    .rev()
                    .map(|arg| format!("\npushq {arg}"))
                    .chain(
                        args.iter()
                            .zip(ARG_REGISTERS.iter())
                            .map(|(arg, reg)| format!("\nmovq {arg}, {reg}")),
                    )
                    .collect::<String>();
                format!(
                    r#"
subq ${args_frame}, {Rsp}
{move_args}
call {symbol}
addq ${args_frame}, {Rsp}
movq {Rax}, {dest}"#,
                    args_frame = 8 * (args.len().saturating_sub(6))
                )
            }
        }
    }
}

#[derive(Clone)]
struct StackOffset(i64);
#[derive(Clone)]
enum Array {
    Stack { offset: StackOffset, size: u32 },
    Global { name: String, size: u32 },
}

impl Array {
    fn size(&self) -> u32 {
        match self {
            Self::Stack { size, .. } => *size,
            Self::Global { size, .. } => *size,
        }
    }
}

impl Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stack { offset, .. } => write!(f, "{}({})", offset.0, Rbp),
            Self::Global { name, .. } => write!(f, "{name}(%rip)"),
        }
    }
}

/// TODO: refactor this thing
#[derive(Clone)]
enum MacVar {
    Stack {
        offset: StackOffset,
    },
    Global {
        name: String,
    },
    /// PERF: this is not the best way to do it
    Index {
        /// offset of the array in the stack!
        base: Array,
        /// offset of the index in the stack!
        index: StackOffset,
    },
    Reg {
        reg: MacReg,
    },
    // well it is not a variable
    Imm {
        value: i64,
    },
}

impl MacVar {
    fn from_source<'a>(
        source: &Source<'a>,
        sym_table: impl FnOnce(Symbol<'a>) -> Option<i64>,
        reg_table: impl FnOnce(u32) -> i64,
        bound_table: impl FnOnce(Symbol<'a>) -> u32,
    ) -> Self {
        match source {
            Source::Immediate(Immediate::Int(value)) => Self::Imm { value: *value },
            Source::Immediate(Immediate::Bool(true)) => Self::Imm { value: 1 },
            Source::Immediate(Immediate::Bool(false)) => Self::Imm { value: 0 },
            &Source::Symbol(s) => {
                if let Some(offset) = sym_table(s) {
                    Self::Stack {
                        offset: StackOffset(offset),
                    }
                } else {
                    Self::Global {
                        name: s.to_string(),
                    }
                }
            }
            Source::Reg(r) => Self::Stack {
                offset: StackOffset(reg_table(r.num())),
            },
            Source::Sc(r) => Self::Stack {
                offset: StackOffset(reg_table(r.num())),
            },
            &Source::Offset(sym, ind) => {
                if let Some(offset) = sym_table(sym) {
                    Self::Index {
                        base: Array::Stack {
                            offset: StackOffset(offset),
                            size: bound_table(sym),
                        },
                        index: StackOffset(reg_table(ind.num())),
                    }
                } else {
                    Self::Index {
                        base: Array::Global {
                            name: sym.to_string(),
                            size: bound_table(sym),
                        },
                        index: StackOffset(reg_table(ind.num())),
                    }
                }
            }
        }
    }

    fn from_dest<'b>(
        dest: &Dest<'b>,
        sym_table: impl FnOnce(Symbol<'b>) -> Option<i64>,
        reg_table: impl FnOnce(u32) -> i64,
        bound_table: impl FnOnce(Symbol<'b>) -> u32,
    ) -> Self {
        match dest {
            &Dest::Symbol(s) => {
                if let Some(offset) = sym_table(s) {
                    Self::Stack {
                        offset: StackOffset(offset),
                    }
                } else {
                    Self::Global {
                        name: s.to_string(),
                    }
                }
            }
            Dest::Reg(r) => Self::Stack {
                offset: StackOffset(reg_table(r.num())),
            },
            Dest::Sc(reg) => Self::Stack {
                offset: StackOffset(reg_table(reg.num())),
            },
            &Dest::Offset(sym, ind) => {
                if let Some(offset) = sym_table(sym) {
                    Self::Index {
                        base: Array::Stack {
                            offset: StackOffset(offset),
                            size: bound_table(sym),
                        },
                        index: StackOffset(reg_table(ind.num())),
                    }
                } else {
                    Self::Index {
                        base: Array::Global {
                            name: sym.to_string(),
                            size: bound_table(sym),
                        },
                        index: StackOffset(reg_table(ind.num())),
                    }
                }
            }
        }
    }
}

impl Display for StackOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({Rbp})", self.0)
    }
}

impl Display for MacVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stack { offset } => write!(f, "{offset}"),
            Self::Reg { reg } => write!(f, "{reg}"),
            Self::Imm { value } => write!(f, "${value}"),
            Self::Global { name } => write!(f, "{name}(%rip)"),
            Self::Index { .. } => unreachable!(),
        }
    }
}

struct MacImmediate(i64);
impl Display for MacImmediate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${imm}", imm = self.0)
    }
}
trait RunTimeError {
    fn sec_label(func: impl Display) -> String;
    fn msg_label(func: impl Display) -> String;
    fn msg(func: impl Display) -> String;
    fn error_code() -> MacImmediate;

    fn codegen(func: impl Display) -> String {
        let func = func.to_string();
        format!(
            r#"
.text
.section .rodata
{msg_label}:
    .string {msg}

.text
{sec_label}:
    {Lea} {msg_label}({Rip}), {Rdi}
    {Mov} $0, {Rax}
    {Call} printf@PLT
    {Mov} {error_code}, {Rdi}
    {Call} exit@PLT"#,
            msg_label = Self::msg_label(&func),
            msg = Self::msg(&func),
            sec_label = Self::sec_label(&func),
            error_code = Self::error_code(),
        )
    }
}

struct BoundCheck;

impl RunTimeError for BoundCheck {
    fn sec_label(func: impl Display) -> String {
        format!("bound_check_{func}")
    }

    fn msg_label(func: impl Display) -> String {
        format!(".bound_check_msg_{func}")
    }

    fn msg(func: impl Display) -> String {
        format!(r#""*** RUNTIME ERROR ***: Array out of Bounds access in method \"{func}\"\n""#)
    }

    fn error_code() -> MacImmediate {
        MacImmediate(1)
    }
}

struct ReturnCheck;
impl RunTimeError for ReturnCheck {
    fn sec_label(func: impl Display) -> String {
        format!("return_check_{func}")
    }

    fn msg_label(func: impl Display) -> String {
        format!(".return_check_msg_{func}")
    }

    fn msg(func: impl Display) -> String {
        format!(r#""*** RUNTIME ERROR ***: No return value from non-void method \"{func}\"\n""#)
    }

    fn error_code() -> MacImmediate {
        MacImmediate(-2)
    }
}

// mod x86 {
//     use std::fmt::Display;
//
//     use crate::ir::{Function, Op, Symbol, Unary};
//
//     use super::{
//         MacReg::{self, *},
//         ARG_REGISTERS,
//     };
//
//     struct Label(String);
//     #[derive(Copy, Clone)]
//     struct LabelRef<'a>(&'a str);
//     #[derive(Copy, Clone)]
//     struct StackOffset(i32);
//
//     #[derive(Copy, Clone)]
//     enum MemVar<'a> {
//         Stack(StackOffset),
//         Global(LabelRef<'a>),
//     }
//
//     impl<'a> MemVar<'a> {
//         fn stack(offset: i32) -> Self {
//             Self::Stack(StackOffset(offset))
//         }
//
//         fn global(label: &'a str) -> Self {
//             Self::Global(LabelRef(label))
//         }
//     }
//
//     impl Display for MemVar<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             match self {
//                 MemVar::Stack(StackOffset(offset)) => write!(f, "{offset}({Rbp})"),
//                 MemVar::Global(LabelRef(label)) => write!(f, "{label}({Rip})"),
//             }
//         }
//     }
//
//     enum SourceDest<'a> {
//         RegReg { source: MacReg, dest: MacReg },
//         RegMem { source: MacReg, dest: MemVar<'a> },
//         MemReg { source: MemVar<'a>, dest: MacReg },
//         ImmReg { source: i64, dest: MacReg },
//         ImmMem { source: i64, dest: MemVar<'a> },
//     }
//
//     impl From<(MacReg, MacReg)> for SourceDest<'_> {
//         fn from(value: (MacReg, MacReg)) -> Self {
//             Self::RegReg {
//                 source: value.0,
//                 dest: value.1,
//             }
//         }
//     }
//
//     impl<'a> From<(MacReg, MemVar<'a>)> for SourceDest<'a> {
//         fn from(value: (MacReg, MemVar<'a>)) -> Self {
//             Self::RegMem {
//                 source: value.0,
//                 dest: value.1,
//             }
//         }
//     }
//
//     impl<'a> From<(MemVar<'a>, MacReg)> for SourceDest<'a> {
//         fn from(value: (MemVar<'a>, MacReg)) -> Self {
//             Self::MemReg {
//                 source: value.0,
//                 dest: value.1,
//             }
//         }
//     }
//
//     impl<'a> From<(i64, MacReg)> for SourceDest<'a> {
//         fn from(value: (i64, MacReg)) -> Self {
//             Self::ImmReg {
//                 source: value.0,
//                 dest: value.1,
//             }
//         }
//     }
//
//     impl<'a> From<(i64, MemVar<'a>)> for SourceDest<'a> {
//         fn from(value: (i64, MemVar<'a>)) -> Self {
//             Self::ImmMem {
//                 source: value.0,
//                 dest: value.1,
//             }
//         }
//     }
//
//     impl<'a> TryFrom<(Source<'a>, Dest<'a>)> for SourceDest<'a> {
//         type Error = ();
//         fn try_from(value: (Source<'a>, Dest<'a>)) -> Result<Self, Self::Error> {
//             match value {
//                 (Source::Reg(source), Dest::Reg(dest)) => Ok(Self::RegReg { source, dest }),
//                 (Source::Reg(source), Dest::Mem(dest)) => Ok(Self::RegMem { source, dest }),
//                 (Source::Mem(source), Dest::Reg(dest)) => Ok(Self::MemReg { source, dest }),
//                 (Source::Imm(source), Dest::Reg(dest)) => Ok(Self::ImmReg { source, dest }),
//                 (Source::Imm(source), Dest::Mem(dest)) => Ok(Self::ImmMem { source, dest }),
//                 _ => Err(()),
//             }
//         }
//     }
//
//     impl<'a> From<(Source<'a>, MacReg)> for SourceDest<'a> {
//         fn from(value: (Source<'a>, MacReg)) -> Self {
//             match value {
//                 (Source::Reg(source), dest) => Self::RegReg { source, dest },
//                 (Source::Mem(source), dest) => Self::MemReg { source, dest },
//                 (Source::Imm(source), dest) => Self::ImmReg { source, dest },
//             }
//         }
//     }
//
//     impl<'a> From<(MacReg, Dest<'a>)> for SourceDest<'a> {
//         fn from(value: (MacReg, Dest<'a>)) -> Self {
//             match value {
//                 (source, Dest::Reg(dest)) => Self::RegReg { source, dest },
//                 (source, Dest::Mem(dest)) => Self::RegMem { source, dest },
//             }
//         }
//     }
//
//     #[derive(Clone, Copy)]
//     enum Source<'a> {
//         Reg(MacReg),
//         Mem(MemVar<'a>),
//         Imm(i64),
//     }
//
//     impl<'a> From<MacReg> for Source<'a> {
//         fn from(value: MacReg) -> Self {
//             Self::Reg(value)
//         }
//     }
//
//     impl Display for Source<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             match self {
//                 Source::Reg(reg) => write!(f, "{reg}"),
//                 Source::Mem(mem) => write!(f, "{mem}"),
//                 Source::Imm(imm) => write!(f, "${imm}"),
//             }
//         }
//     }
//
//     #[derive(Copy, Clone)]
//     enum Dest<'a> {
//         Reg(MacReg),
//         Mem(MemVar<'a>),
//     }
//
//     impl Display for SourceDest<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             match self {
//                 SourceDest::RegReg { source, dest } => write!(f, "{source}, {dest}"),
//                 SourceDest::RegMem { source, dest } => write!(f, "{source}, {dest}"),
//                 SourceDest::MemReg { source, dest } => write!(f, "{source}, {dest}"),
//                 SourceDest::ImmReg { source, dest } => write!(f, "${source}, {dest}"),
//                 SourceDest::ImmMem { source, dest } => write!(f, "${source}, {dest}"),
//             }
//         }
//     }
//
//     enum Instruction<'a> {
//         Mov(SourceDest<'a>),
//         Add(SourceDest<'a>),
//         Sub(SourceDest<'a>),
//         And(SourceDest<'a>),
//         Or(SourceDest<'a>),
//         Xor(SourceDest<'a>),
//         IDiv(Source<'a>),
//         IMul(Source<'a>),
//         Neg(MacReg),
//         Cmp(MacReg, MacReg),
//         Cmov(SourceDest<'a>),
//         Cmovl(SourceDest<'a>),
//         Cmovle(SourceDest<'a>),
//         Cmovg(SourceDest<'a>),
//         Cmovge(SourceDest<'a>),
//         Cmovz(SourceDest<'a>),
//         Cmovnz(SourceDest<'a>),
//         Cqto,
//         Leave,
//         Ret,
//     }
//
//     impl<'a> Instruction<'a> {
//         fn cmove(op: Op, source_dest: SourceDest<'a>) -> Self {
//             match op {
//                 Op::Equal => Self::Cmovz(source_dest),
//                 Op::NotEqual => Self::Cmovnz(source_dest),
//                 Op::Less => Self::Cmovl(source_dest),
//                 Op::LessEqual => Self::Cmovle(source_dest),
//                 Op::Greater => Self::Cmovg(source_dest),
//                 Op::GreaterEqual => Self::Cmovge(source_dest),
//                 _ => unreachable!(),
//             }
//         }
//
//         /// returns a bin op instruction (if it is supported for example it can not be used with
//         /// multiplication and division). otherwise it panics.
//         fn binop(op: Op, source_dest: SourceDest<'a>) -> Self {
//             match op {
//                 Op::Add => Self::Add(source_dest),
//                 Op::Sub => Self::Sub(source_dest),
//                 Op::And => Self::And(source_dest),
//                 Op::Or => Self::Or(source_dest),
//                 op => unreachable!(
//                     "this operation can not be expressed with a single machine instruction: {op:?}"
//                 ),
//             }
//         }
//     }
//
//     impl Display for Instruction<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             match self {
//                 Instruction::Mov(source_dest) => write!(f, "movq {source_dest}"),
//                 Instruction::Add(source_dest) => write!(f, "addq {source_dest}"),
//                 Instruction::Sub(source_dest) => write!(f, "subq {source_dest}"),
//                 Instruction::And(source_dest) => write!(f, "andq {source_dest}"),
//                 Instruction::Or(source_dest) => write!(f, "orq {source_dest}"),
//                 Instruction::Xor(source_dest) => write!(f, "xorq {source_dest}"),
//                 Instruction::Cmov(source_dest) => write!(f, "cmoveq {source_dest}"),
//                 Instruction::Cmovl(source_dest) => write!(f, "cmovl {source_dest}"),
//                 Instruction::Cmovle(source_dest) => write!(f, "cmovle {source_dest}"),
//                 Instruction::Cmovg(source_dest) => write!(f, "cmovg {source_dest}"),
//                 Instruction::Cmovge(source_dest) => write!(f, "cmovge {source_dest}"),
//                 Instruction::Cmovz(source_dest) => write!(f, "cmovz {source_dest}"),
//                 Instruction::Cmovnz(source_dest) => write!(f, "cmovnz {source_dest}"),
//                 Instruction::Cmp(reg1, reg2) => write!(f, "cmp {reg1}, {reg2}"),
//                 Instruction::Neg(reg) => write!(f, "negq {reg}"),
//                 Instruction::IMul(source) => write!(f, "imulq {source}"),
//                 Instruction::IDiv(source) => write!(f, "idivq {source}"),
//                 Instruction::Cqto => write!(f, "cqto"),
//                 Instruction::Leave => write!(f, "leave"),
//                 Instruction::Ret => write!(f, "ret"),
//             }
//         }
//     }
//
//     struct StringDef<'a> {
//         label: LabelRef<'a>,
//         string: String,
//     }
//
//     impl Display for StringDef<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             write!(
//                 f,
//                 r#"{label}:
//   .string {string}"#,
//                 label = self.label.0,
//                 string = self.string
//             )
//         }
//     }
//
//     struct GlobalDef<'a> {
//         name: &'a str,
//         size: usize,
//     }
//
//     impl Display for GlobalDef<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             write!(
//                 f,
//                 r#".globl {name}
// .align 8
// .size {name}, {size}
// {name}:
//   .zero {size}"#,
//                 name = self.name,
//                 size = self.size
//             )
//         }
//     }
//
//     struct Section<'a> {
//         label: Label,
//         instructions: Vec<Instruction<'a>>,
//     }
//
//     impl<'a> Section<'a> {
//         fn new(label: Label) -> Self {
//             Self {
//                 label,
//                 instructions: Vec::new(),
//             }
//         }
//
//         fn add_instruction(&mut self, instruction: Instruction<'a>) -> &mut Self {
//             self.instructions.push(instruction);
//             self
//         }
//     }
//
//     impl Display for Section<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             writeln!(f, "{label}:", label = self.label.0)?;
//             self.instructions
//                 .iter()
//                 .try_fold((), |_, instruction| writeln!(f, "  {instruction}"))
//         }
//     }
//
//     struct AsmFunction<'a> {
//         // we do not need to store the name of the function since it is stored in the first
//         // section.
//         sections: Vec<Section<'a>>,
//     }
//
//     impl Display for AsmFunction<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             writeln!(f, ".text")?;
//             writeln!(f, ".globl {func}", func = self.sections[0].label.0)?;
//             self.sections
//                 .iter()
//                 .try_fold((), |_, section| writeln!(f, "{section}"))
//         }
//     }
//
//     #[derive(Default)]
//     struct AsmArchive<'a> {
//         ro_data: Vec<StringDef<'a>>,
//         bss_data: Vec<GlobalDef<'a>>,
//         functions: Vec<AsmFunction<'a>>,
//     }
//
//     impl<'a> AsmArchive<'a> {
//         pub fn new() -> Self {
//             Self::default()
//         }
//
//         pub fn write_string_def(&mut self, string_def: StringDef<'a>) {
//             self.ro_data.push(string_def);
//         }
//
//         pub fn write_global_def(&mut self, global_def: GlobalDef<'a>) {
//             self.bss_data.push(global_def);
//         }
//
//         pub fn write_function(&mut self, function: AsmFunction<'a>) {
//             self.functions.push(function);
//         }
//     }
//
//     impl Display for AsmArchive<'_> {
//         fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//             writeln!(f, ".text")?;
//             writeln!(f, ".section .rodata")?;
//             self.ro_data
//                 .iter()
//                 .try_fold((), |_, string_def| writeln!(f, "{string_def}"))?;
//
//             writeln!(f, ".text")?;
//             writeln!(f, ".section .bss")?;
//             self.bss_data
//                 .iter()
//                 .try_fold((), |_, global_def| writeln!(f, "{global_def}"))?;
//
//             writeln!(f, ".text")?;
//             self.functions
//                 .iter()
//                 .try_fold((), |_, function| writeln!(f, "{function}"))
//         }
//     }
//
//     use crate::ir::{Dest as IRDest, Instruction as IRInstruction, Source as IRSource};
//
//     impl<'a> IRInstruction<'_> {
//         fn to_macinstructions(
//             &self,
//             section: &mut Section<'a>,
//             mut irsource_to_source: impl FnMut(&IRSource) -> Source<'a>,
//             irdest_to_dest: impl FnOnce(&IRDest) -> Dest<'a>,
//         ) {
//             match self {
//                 Self::AllocArray { .. } | Self::AllocScalar { .. } => {}
//                 Self::Load { dest, source } | Self::Store { dest, source } => {
//                     let source = irsource_to_source(source);
//                     let dest = irdest_to_dest(dest);
//                     if let Ok(sourcedest) = SourceDest::try_from((source, dest)) {
//                         section.add_instruction(Instruction::Mov(sourcedest));
//                     } else if let (Source::Mem(source), Dest::Mem(dest)) = (source, dest) {
//                         section
//                             .add_instruction(Instruction::Mov(SourceDest::MemReg {
//                                 source,
//                                 dest: R10,
//                             }))
//                             .add_instruction(Instruction::Mov(SourceDest::RegMem {
//                                 source: R10,
//                                 dest,
//                             }));
//                     } else {
//                         unreachable!()
//                     }
//                 }
//                 Self::Return { value } => {
//                     let source = irsource_to_source(value);
//                     section
//                         .add_instruction(Instruction::Mov(SourceDest::from((source, Rax))))
//                         .add_instruction(Instruction::Leave)
//                         .add_instruction(Instruction::Ret);
//                 }
//                 Self::VoidReturn => {
//                     section
//                         .add_instruction(Instruction::Leave)
//                         .add_instruction(Instruction::Ret);
//                 }
//                 Self::Unary {
//                     dest,
//                     source,
//                     op: Unary::Not,
//                 } => {
//                     let source = irsource_to_source(source);
//                     let dest = irdest_to_dest(dest);
//                     section
//                         .add_instruction(Instruction::Mov(SourceDest::from((source, R10))))
//                         .add_instruction(Instruction::Mov(SourceDest::from((1, R11))))
//                         .add_instruction(Instruction::Xor(SourceDest::from((R10, R11))))
//                         .add_instruction(Instruction::Mov(SourceDest::from((R11, dest))));
//                 }
//                 Self::Unary {
//                     dest,
//                     source,
//                     op: Unary::Neg,
//                 } => {
//                     let source = irsource_to_source(source);
//                     let dest = irdest_to_dest(dest);
//                     section
//                         .add_instruction(Instruction::Mov(SourceDest::from((source, R10))))
//                         .add_instruction(Instruction::Neg(R10))
//                         .add_instruction(Instruction::Mov(SourceDest::from((R10, dest))));
//                 }
//                 IRInstruction::Select {
//                     dest,
//                     cond,
//                     yes,
//                     no,
//                 } => {
//                     let cond = irsource_to_source(cond);
//                     let yes = irsource_to_source(yes);
//                     let no = irsource_to_source(no);
//                     let dest = irdest_to_dest(dest);
//                     section
//                         .add_instruction(Instruction::Mov(SourceDest::from((cond, R10))))
//                         .add_instruction(Instruction::Mov(SourceDest::from((1, R11))))
//                         .add_instruction(Instruction::Cmp(R10, R11))
//                         .add_instruction(Instruction::Mov(SourceDest::from((yes, R10))))
//                         .add_instruction(Instruction::Mov(SourceDest::from((no, R11))))
//                         .add_instruction(Instruction::Cmov(SourceDest::from((R10, R11))))
//                         .add_instruction(Instruction::Mov(SourceDest::from((R11, dest))));
//                 }
//                 Self::Op2 { dest, lhs, rhs, op } => {
//                     let lhs = irsource_to_source(lhs);
//                     let rhs = irsource_to_source(rhs);
//                     let dest = irdest_to_dest(dest);
//                     if let Op::Mul = op {
//                         section
//                             .add_instruction(Instruction::Mov(SourceDest::from((lhs, Rax))))
//                             .add_instruction(Instruction::IMul(rhs))
//                             .add_instruction(Instruction::Mov(SourceDest::from((Rax, dest))));
//                     } else if let Op::Div = op {
//                         section
//                             .add_instruction(Instruction::Mov(SourceDest::from((lhs, Rax))))
//                             .add_instruction(Instruction::Cqto)
//                             .add_instruction(Instruction::IDiv(rhs))
//                             .add_instruction(Instruction::Mov(SourceDest::from((Rax, dest))));
//                     } else if let Op::Mod = op {
//                         section
//                             .add_instruction(Instruction::Mov(SourceDest::from((lhs, Rax))))
//                             .add_instruction(Instruction::Cqto)
//                             .add_instruction(Instruction::IDiv(rhs))
//                             .add_instruction(Instruction::Mov(SourceDest::from((Rdx, dest))));
//                     } else if op.is_cmp() {
//                         section
//                             .add_instruction(Instruction::Mov(SourceDest::from((lhs, R10))))
//                             .add_instruction(Instruction::Mov(SourceDest::from((rhs, R11))))
//                             .add_instruction(Instruction::Cmp(R11, R10))
//                             .add_instruction(Instruction::Mov(SourceDest::from((1, R10))))
//                             .add_instruction(Instruction::Mov(SourceDest::from((0, R11))))
//                             .add_instruction(Instruction::cmove(*op, SourceDest::from((R10, R11))))
//                             .add_instruction(Instruction::Mov(SourceDest::from((R11, dest))));
//                     } else if op.is_arith() || op.is_logic() {
//                         section
//                             .add_instruction(Instruction::Mov(SourceDest::from((lhs, R11))))
//                             .add_instruction(Instruction::Mov(SourceDest::from((rhs, R10))))
//                             .add_instruction(Instruction::binop(*op, SourceDest::from((R10, R11))))
//                             .add_instruction(Instruction::Mov(SourceDest::from((R11, dest))));
//                     }
//                 }
//             }
//         }
//     }
//
//     impl Function<'_> {
//         fn wip_codegen(&self, archive: &mut AsmArchive) {}
//
//         fn move_args(&self) -> impl Iterator<Item = Instruction> {
//             self.args()
//                 .iter()
//                 .enumerate()
//                 .zip(ARG_REGISTERS.iter())
//                 .map(|((i, _), &reg)| {
//                     Instruction::Mov(SourceDest::RegMem {
//                         source: reg,
//                         dest: MemVar::Stack(StackOffset(i as i32 * -8)),
//                     })
//                 })
//         }
//     }
// }

impl Function<'_> {
    /// generates the x86_64 code for the function.
    pub fn codegen(
        &self,
        const_strings: &mut usize,
        mut global_bound_tale: impl FnMut(Symbol) -> u32,
    ) -> (String, String) {
        let mut symbols_to_offset = self
            .args()
            .iter()
            .take(6)
            .enumerate()
            .map(|(i, arg)| (*arg, -(i as i64 + 1) * 8))
            .collect::<HashMap<_, _>>();
        let mut stack_top = symbols_to_offset.len() as i64 * -8;
        symbols_to_offset.extend(
            self.args()
                .iter()
                .skip(6)
                .enumerate()
                .map(|(i, arg)| (*arg, (i as i64 + 2) * 8)),
        );

        let mut bound_table = HashMap::new();

        symbols_to_offset.extend(self.graph().dfs().flat_map(|node| {
            node.as_ref()
                .insrtuctions()
                .iter()
                .filter_map(|instruction| match *instruction {
                    Instruction::AllocScalar { name, .. } => Some((name, {
                        stack_top -= 8;
                        stack_top
                    })),
                    Instruction::AllocArray { name, size } => Some((name, {
                        stack_top -= (size * 8) as i64;
                        bound_table.insert(name, (size) as u32);
                        stack_top
                    })),
                    _ => None,
                })
                .collect::<Vec<_>>()
        }));

        let mut const_defs = String::default();
        let func_asm = self
            .args()
            .iter()
            .zip(ARG_REGISTERS.iter())
            .map(|(sym, &reg)| LIr::Mov {
                dest: MacVar::Stack {
                    offset: StackOffset(*symbols_to_offset.get(sym).unwrap()),
                },
                source: MacVar::Reg { reg },
            })
            .map(|lir| {
                lir.codegen(
                    || BoundCheck::sec_label(self.name()),
                    || BoundCheck::sec_label(self.name()),
                )
            })
            .chain(self.graph().psudo_inorder().map(|node| {
                let reg_table = |reg: u32| (reg as i64 + 1) * -8 + stack_top;
                let linear_instructions = node
                    .as_ref()
                    .insrtuctions()
                    .iter()
                    .filter_map(|instruction| {
                        LIr::from_ir(
                            instruction,
                            |sym| symbols_to_offset.get(&sym).copied(),
                            reg_table,
                            |sym| {
                                bound_table
                                    .get(&sym)
                                    .copied()
                                    .unwrap_or_else(|| global_bound_tale(sym))
                            },
                            const_strings,
                        )
                        .map(|(lir, const_def)| {
                            if let Some(def) = const_def {
                                const_defs.push_str(&def)
                            }
                            lir.codegen(
                                || BoundCheck::sec_label(self.name()),
                                || BoundCheck::sec_label(self.name()),
                            )
                        })
                    })
                    .collect::<String>();
                let terminator = match node.neighbours() {
                    None => String::default(),
                    Some(NeighbouringNodes::Unconditional { next }) => {
                        format!("\njmp {}", next.id())
                    }
                    Some(NeighbouringNodes::Conditional { yes, no, cond }) => {
                        format!(
                            r#"
movq {cond}, {R10}
movq $1, {R11}
cmp {R10}, {R11}
je {yes}
jmp {no}"#,
                            cond = MacVar::from_source(
                                &Source::from(cond),
                                |_| unreachable!(),
                                reg_table,
                                |sym| bound_table
                                    .get(&sym)
                                    .copied()
                                    .unwrap_or_else(|| global_bound_tale(sym)),
                            ),
                            no = no.id(),
                            yes = yes.id()
                        )
                    }
                };
                format!(
                    "\n{label}:\n{linear_instructions}\n{terminator}",
                    label = node.id()
                )
            }))
            .collect::<String>();

        let stack_length = (-stack_top) as u32 + 8 * self.allocated_regs();

        (
            format!(
                r#"
                .globl {name}
                {name}:
                enter $({stack_length}), $0
                {func_asm}
                "#,
                name = self.name()
            ),
            format!(
                r#"
                    {const_defs}
                    {bound_runtime_error}
                    {no_return_runtime_error}
                    "#,
                bound_runtime_error = BoundCheck::codegen(self.name()),
                no_return_runtime_error = ReturnCheck::codegen(self.name()),
            ),
        )
    }
}

impl Program<'_> {
    pub fn codegen(&self) -> String {
        let mut const_strings = 0;
        let bss = self
            .globals()
            .iter()
            .map(|global| {
                format!(
                    r#"
.text
.globl {name}
.bss
.align 8
.size {name}, {size}
{name}:
.zero {size}
"#,
                    name = global.0,
                    size = global.1,
                )
            })
            .collect::<String>();
        let globals_bounds = self
            .globals()
            .iter()
            .filter_map(|(sym, size)| {
                if *size > 8 {
                    Some((sym, (*size) as u32 / 8))
                } else {
                    None
                }
            })
            .collect::<HashMap<_, _>>();
        let (functions, ro_data) = self
            .functions()
            .iter()
            .map(|func| {
                func.codegen(&mut const_strings, |sym| {
                    globals_bounds.get(&sym).copied().unwrap()
                })
            })
            .fold(
                (String::new(), String::new()),
                |(mut functions, mut ro_data), (function_new, ro_data_new)| {
                    functions.push_str(&function_new);
                    ro_data.push_str(&ro_data_new);
                    (functions, ro_data)
                },
            );
        format!(
            r#"{bss}
.text
.section .rodata
{ro_data}
.text
{functions}"#
        )
    }
}
