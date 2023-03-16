use crate::ir::{
    Dest, Function, IRExternArg, Immediate, Instruction, NeighbouringNodes, Op, Program, Reg,
    Source, Symbol, Unary,
};
use std::{collections::HashMap, fmt::Display};

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
            op => todo!("{op:?}"),
        }
    }
}

use MacReg::*;

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

enum LIr {
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
        symbol: Symbol,
        args: Vec<MacVar>,
    },
    Call {
        dest: MacVar,
        symbol: Symbol,
        args: Vec<MacVar>,
    },
    ExternCall {
        dest: MacVar,
        symbol: Symbol,
        args: Vec<ExternArg>,
    },
    Return {
        value: MacVar,
    },
    VoidReturn,
}

impl LIr {
    fn from_ir(
        instruction: &Instruction,
        sym_table: impl Fn(&Symbol) -> Option<i64>,
        reg_table: impl Fn(Reg) -> i64,
        mut bound_table: impl FnMut(&Symbol) -> u32,
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
            Instruction::Load { dest, source } | Instruction::Store { dest, source } => Some((
                Self::Mov {
                    dest: MacVar::from_dest(dest, &sym_table, &reg_table, &mut bound_table),
                    source: MacVar::from_source(source, &sym_table, &reg_table, &mut bound_table),
                },
                None,
            )),
            Instruction::Op2 {
                dest,
                source1,
                source2,
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
                        symbol: symbol.to_string(),
                        args,
                    },
                    Some(const_defs),
                )
            }),

            Instruction::ReturnGuard => Some((Self::ReturnGuard, None)),
            Instruction::VoidReturn => Some((Self::VoidReturn, None)),
            Instruction::VoidCall { symbol, args } => Some((
                Self::VoidCall {
                    symbol: symbol.to_string(),
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
                    symbol: symbol.to_string(),
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
    fn codegen(&self, on_bound_fail: impl FnOnce() -> String) -> String {
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
movq $-2, {Rdi}
call exit@PLT
"#
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
    fn from_source(
        source: &Source,
        sym_table: impl FnOnce(&Symbol) -> Option<i64>,
        reg_table: impl FnOnce(Reg) -> i64,
        bound_table: impl FnOnce(&Symbol) -> u32,
    ) -> Self {
        match source {
            Source::Immediate(Immediate::Int(value)) => Self::Imm { value: *value },
            Source::Immediate(Immediate::Bool(true)) => Self::Imm { value: 1 },
            Source::Immediate(Immediate::Bool(false)) => Self::Imm { value: 0 },
            Source::Symbol(s) => {
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
                offset: StackOffset(reg_table(*r)),
            },
            Source::Offset(sym, ind) => {
                if let Some(offset) = sym_table(sym) {
                    Self::Index {
                        base: Array::Stack {
                            offset: StackOffset(offset),
                            size: bound_table(sym),
                        },
                        index: StackOffset(reg_table(*ind)),
                    }
                } else {
                    Self::Index {
                        base: Array::Global {
                            name: sym.to_string(),
                            size: bound_table(sym),
                        },
                        index: StackOffset(reg_table(*ind)),
                    }
                }
            }
        }
    }

    fn from_dest(
        dest: &Dest,
        sym_table: impl FnOnce(&Symbol) -> Option<i64>,
        reg_table: impl FnOnce(Reg) -> i64,
        bound_table: impl FnOnce(&Symbol) -> u32,
    ) -> Self {
        match dest {
            Dest::Symbol(s) => {
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
                offset: StackOffset(reg_table(*r)),
            },
            Dest::Offset(sym, ind) => {
                if let Some(offset) = sym_table(sym) {
                    Self::Index {
                        base: Array::Stack {
                            offset: StackOffset(offset),
                            size: bound_table(sym),
                        },
                        index: StackOffset(reg_table(*ind)),
                    }
                } else {
                    Self::Index {
                        base: Array::Global {
                            name: sym.to_string(),
                            size: bound_table(sym),
                        },
                        index: StackOffset(reg_table(*ind)),
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

impl Function {
    /// generates the x86_64 code for the function.
    pub fn codegen(
        &self,
        const_strings: &mut usize,
        mut global_bound_tale: impl FnMut(&String) -> u32,
    ) -> (String, String) {
        // move the arguments to the stack

        let mut symbols_to_offset = self
            .args()
            .iter()
            .take(6)
            .enumerate()
            .map(|(i, arg)| (arg.clone(), -(i as i64 + 1) * 8))
            .collect::<HashMap<_, _>>();
        let mut stack_top = symbols_to_offset.len() as i64 * -8;
        symbols_to_offset.extend(
            self.args()
                .iter()
                .skip(6)
                .enumerate()
                .map(|(i, arg)| (arg.clone(), (i as i64 + 2) * 8)),
        );

        let mut bound_table = HashMap::new();

        symbols_to_offset.extend(self.graph().dfs().flat_map(|node| {
            node.as_ref()
                .insrtuctions()
                .iter()
                .filter_map(|instruction| match instruction {
                    Instruction::AllocScalar { name, .. } => Some((name.clone(), {
                        stack_top -= 8;
                        stack_top
                    })),
                    Instruction::AllocArray { name, size } => Some((name.clone(), {
                        stack_top -= (*size * 8) as i64;
                        bound_table.insert(name.clone(), (*size) as u32);
                        stack_top
                    })),
                    _ => None,
                })
                .collect::<Vec<_>>()
        }));

        let out_of_bound_sec = format!("out_of_bound_exit_{func_name}", func_name = self.name());
        let out_of_bound_msg_label = format!(".out_of_bound_{func_name}", func_name = self.name());
        let out_of_bound_msg = format!(
            r#""*** RUNTIME ERROR ***: Array out of Bounds access in method \"{func_name}\"\n""#,
            func_name = self.name()
        );
        let out_of_bound_str = format!(
            r#"
{sym}:
    .string {string}
"#,
            sym = out_of_bound_msg_label,
            string = out_of_bound_msg
        );
        let exit_prog = format!(
            r#"
.text
{sec}:
leaq {msg}(%rip), %rdi
movq $0, %rax
call printf@PLT
movq $-1, %rdi
call exit@PLT
"#,
            sec = out_of_bound_sec,
            msg = out_of_bound_msg_label
        );
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
            .map(|lir| lir.codegen(|| out_of_bound_sec.clone()))
            .chain(self.graph().psudo_inorder().map(|node| {
                let reg_table = |reg: Reg| (reg.num() as i64 + 1) * -8 + stack_top;
                let linear_instructions = node
                    .as_ref()
                    .insrtuctions()
                    .iter()
                    .filter_map(|instruction| {
                        LIr::from_ir(
                            instruction,
                            |sym| symbols_to_offset.get(sym).copied(),
                            reg_table,
                            |sym| {
                                bound_table
                                    .get(sym)
                                    .copied()
                                    .unwrap_or_else(|| global_bound_tale(sym))
                            },
                            const_strings,
                        )
                        .map(|(lir, const_def)| {
                            if let Some(def) = const_def {
                                const_defs.push_str(&def)
                            }
                            lir.codegen(|| out_of_bound_sec.clone())
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
                                    .get(sym)
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

        let stack_length = (-stack_top) as usize + 8 * self.allocated_regs();

        if self.name() == "main" {
            (
                format!(
                    r#"
.globl main
main:
enter $({stack_length}), $0
{func_asm}
"#,
                ),
                format!("{const_defs}\n{out_of_bound_str}\n{exit_prog}\n"),
            )
        } else {
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
                format!("{const_defs}\n{out_of_bound_str}\n{exit_prog}\n"),
            )
        }
    }
}

impl Program {
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
                    globals_bounds.get(sym).copied().unwrap()
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
