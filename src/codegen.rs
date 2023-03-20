use crate::ir::{Function, IRExternArg, NeighbouringNodes, NodeId, Op, Program, Symbol, Unary};
use std::{collections::HashMap, fmt::Display};

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

impl Display for StackOffset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({Rbp})", self.0)
    }
}

struct MacImmediate(i64);
impl Display for MacImmediate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "${imm}", imm = self.0)
    }
}

struct NonFunctionLabel(NodeId);

impl Display for NonFunctionLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Copy, Clone)]
struct LabelRef<'a>(&'a str);

#[derive(Copy, Clone, Debug)]
struct StackOffset(i32);

#[derive(Copy, Clone)]
struct Global {
    id: u16,
    size: usize,
}

impl Display for Global {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#"
.align 8
.size {id}, {size}
{id}:
  .zero {size}"#,
            id = GlobalRef(self.id),
            size = self.size
        )
    }
}

#[derive(Copy, Clone, Debug)]
struct GlobalRef(u16);

impl GlobalRef {
    fn label(&self) -> String {
        format!("G{}", self.0)
    }
}

impl Display for GlobalRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{label}", label = self.label())
    }
}

#[derive(Copy, Clone, Debug)]
enum MemVar {
    Stack(StackOffset),
    ReadOnly(StringRef),
    Global(GlobalRef),
}

impl From<StringRef> for MemVar {
    fn from(value: StringRef) -> Self {
        Self::ReadOnly(value)
    }
}

impl Display for MemVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemVar::Stack(StackOffset(offset)) => write!(f, "{offset}({Rbp})"),
            MemVar::Global(global) => write!(f, "{global}({Rip})"),
            MemVar::ReadOnly(label) => write!(f, "{label}({Rip})"),
        }
    }
}

enum SourceDest {
    RegReg { source: MacReg, dest: MacReg },
    RegMem { source: MacReg, dest: MemVar },
    MemReg { source: MemVar, dest: MacReg },
    ImmReg { source: i64, dest: MacReg },
    ImmMem { source: i64, dest: MemVar },
    IndReg { source: Index, dest: MacReg },
    RegInd { source: MacReg, dest: Index },
}

impl From<(MacReg, MacReg)> for SourceDest {
    fn from(value: (MacReg, MacReg)) -> Self {
        Self::RegReg {
            source: value.0,
            dest: value.1,
        }
    }
}

impl From<(MacReg, MemVar)> for SourceDest {
    fn from(value: (MacReg, MemVar)) -> Self {
        Self::RegMem {
            source: value.0,
            dest: value.1,
        }
    }
}

impl From<(MemVar, MacReg)> for SourceDest {
    fn from(value: (MemVar, MacReg)) -> Self {
        Self::MemReg {
            source: value.0,
            dest: value.1,
        }
    }
}

impl From<(i64, MacReg)> for SourceDest {
    fn from(value: (i64, MacReg)) -> Self {
        Self::ImmReg {
            source: value.0,
            dest: value.1,
        }
    }
}

impl From<(i64, MemVar)> for SourceDest {
    fn from(value: (i64, MemVar)) -> Self {
        Self::ImmMem {
            source: value.0,
            dest: value.1,
        }
    }
}

impl From<(Source, MacReg)> for SourceDest {
    fn from(value: (Source, MacReg)) -> Self {
        match value {
            (Source::Reg(source), dest) => Self::RegReg { source, dest },
            // (Source::Mem(source), dest) => Self::MemReg { source, dest },
            // (Source::Imm(source), dest) => Self::ImmReg { source, dest },
            // (Source::Index(source), dest) => Self::IndReg { source, dest },
        }
    }
}

impl From<(MacReg, Dest)> for SourceDest {
    fn from(value: (MacReg, Dest)) -> Self {
        match value {
            (source, Dest::Reg(dest)) => Self::RegReg { source, dest },
            (source, Dest::Mem(dest)) => Self::RegMem { source, dest },
            // (source, Dest::Index(dest)) => Self::RegInd { source, dest },
        }
    }
}

#[derive(Clone, Copy)]
struct Index {
    base: MacReg,
    offset: MacReg,
}

impl Display for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({base}, {offset}, {size})",
            offset = self.offset,
            base = self.base,
            size = 8,
        )
    }
}

#[derive(Clone, Copy)]
enum Source {
    Reg(MacReg),
    // Mem(MemVar),
    // Imm(i64),
    // Index(Index),
}

impl From<MacReg> for Source {
    fn from(value: MacReg) -> Self {
        Self::Reg(value)
    }
}

impl Display for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::Reg(reg) => write!(f, "{reg}"),
            // Source::Mem(mem) => write!(f, "{mem}"),
            // Source::Imm(imm) => write!(f, "${imm}"),
            // Source::Index(ind) => write!(f, "{ind}"),
        }
    }
}

#[derive(Copy, Clone)]
enum Dest {
    Reg(MacReg),
    Mem(MemVar),
    // Index(Index),
}

impl From<MemVar> for Dest {
    fn from(value: MemVar) -> Self {
        Self::Mem(value)
    }
}

impl Display for Dest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reg(reg) => write!(f, "{reg}"),
            Self::Mem(mem) => write!(f, "{mem}"),
            // Self::Index(ind) => write!(f, "{ind}"),
        }
    }
}

impl From<MacReg> for Dest {
    fn from(value: MacReg) -> Self {
        Self::Reg(value)
    }
}

impl Display for SourceDest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceDest::RegReg { source, dest } => write!(f, "{source}, {dest}"),
            SourceDest::RegMem { source, dest } => write!(f, "{source}, {dest}"),
            SourceDest::MemReg { source, dest } => write!(f, "{source}, {dest}"),
            SourceDest::ImmReg { source, dest } => write!(f, "${source}, {dest}"),
            SourceDest::ImmMem { source, dest } => write!(f, "${source}, {dest}"),
            SourceDest::RegInd { source, dest } => write!(f, "{source}, {dest}"),
            SourceDest::IndReg { source, dest } => write!(f, "{source}, {dest}"),
        }
    }
}

enum Instruction<'a> {
    Mov(SourceDest),
    Add(SourceDest),
    Sub(SourceDest),
    And(SourceDest),
    Or(SourceDest),
    Xor(SourceDest),
    IDiv(Source),
    IMul(Source),
    Neg(MacReg),
    Cmp(MacReg, MacReg),
    Test(MacReg, MacReg),
    Cmov(SourceDest),
    Cmovl(SourceDest),
    Cmovle(SourceDest),
    Cmovg(SourceDest),
    Cmovge(SourceDest),
    Cmovz(SourceDest),
    Cmovnz(SourceDest),
    Push(MacReg),
    Call(&'a str),
    Leaq(MemVar, MacReg),
    Jmp(NonFunctionLabel),
    Jz(NonFunctionLabel),
    Enter(usize),
    Cqto,
    Leave,
    Ret,
}

impl<'a> Instruction<'a> {
    fn cmove(op: Op, source_dest: SourceDest) -> Self {
        match op {
            Op::Equal => Self::Cmovz(source_dest),
            Op::NotEqual => Self::Cmovnz(source_dest),
            Op::Less => Self::Cmovl(source_dest),
            Op::LessEqual => Self::Cmovle(source_dest),
            Op::Greater => Self::Cmovg(source_dest),
            Op::GreaterEqual => Self::Cmovge(source_dest),
            _ => unreachable!(),
        }
    }

    /// returns a bin op instruction (if it is supported for example it can not be used with
    /// multiplication and division). otherwise it panics.
    fn binop(op: Op, source_dest: SourceDest) -> Self {
        match op {
            Op::Add => Self::Add(source_dest),
            Op::Sub => Self::Sub(source_dest),
            Op::And => Self::And(source_dest),
            Op::Or => Self::Or(source_dest),
            op => unreachable!(
                "this operation can not be expressed with a single machine instruction: {op:?}"
            ),
        }
    }
}

impl Display for Instruction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Mov(source_dest) => write!(f, "movq {source_dest}"),
            Instruction::Add(source_dest) => write!(f, "addq {source_dest}"),
            Instruction::Sub(source_dest) => write!(f, "subq {source_dest}"),
            Instruction::And(source_dest) => write!(f, "andq {source_dest}"),
            Instruction::Or(source_dest) => write!(f, "orq {source_dest}"),
            Instruction::Xor(source_dest) => write!(f, "xorq {source_dest}"),
            Instruction::Cmov(source_dest) => write!(f, "cmoveq {source_dest}"),
            Instruction::Cmovl(source_dest) => write!(f, "cmovl {source_dest}"),
            Instruction::Cmovle(source_dest) => write!(f, "cmovle {source_dest}"),
            Instruction::Cmovg(source_dest) => write!(f, "cmovg {source_dest}"),
            Instruction::Cmovge(source_dest) => write!(f, "cmovge {source_dest}"),
            Instruction::Cmovz(source_dest) => write!(f, "cmovz {source_dest}"),
            Instruction::Cmovnz(source_dest) => write!(f, "cmovnz {source_dest}"),
            Instruction::Enter(size) => write!(f, "enter ${size}, $0"),
            Instruction::Cmp(reg1, reg2) => write!(f, "cmp {reg1}, {reg2}"),
            Instruction::Test(reg1, reg2) => write!(f, "test {reg1}, {reg2}"),
            Instruction::Neg(reg) => write!(f, "negq {reg}"),
            Instruction::IMul(source) => write!(f, "imulq {source}"),
            Instruction::IDiv(source) => write!(f, "idivq {source}"),
            Instruction::Call(name) => write!(f, "call {name}"),
            Instruction::Jmp(label) => write!(f, "jmp {label}"),
            Instruction::Jz(label) => write!(f, "jz {label}"),
            Instruction::Push(source) => write!(f, "pushq {source}"),
            Instruction::Leaq(mem, dest) => write!(f, "leaq {mem}, {dest}"),
            Instruction::Cqto => write!(f, "cqto"),
            Instruction::Leave => write!(f, "leave"),
            Instruction::Ret => write!(f, "ret"),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
struct StringDef<'a> {
    string: &'a str,
    id: u16,
}

impl StringDef<'_> {
    fn get_ref(&self) -> StringRef {
        StringRef(self.id)
    }
}

#[derive(Debug, Clone, Copy)]
struct StringRef(u16);

impl Display for StringRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, ".CStr{}", self.0)
    }
}

impl Display for StringDef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#".CStr{id}:
  .string "{string}""#,
            id = self.id,
            string = self.string
        )
    }
}

struct GlobalDef<'a> {
    name: &'a str,
    size: usize,
}

impl Display for GlobalDef<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            r#".globl {name}
.align 8
.size {name}, {size}
{name}:
  .zero {size}"#,
            name = self.name,
            size = self.size
        )
    }
}

struct Section<'a> {
    label: NonFunctionLabel,
    instructions: Vec<Instruction<'a>>,
}

impl<'a> Section<'a> {
    fn new(label: NonFunctionLabel) -> Self {
        Self {
            label,
            instructions: Vec::new(),
        }
    }

    fn add_instruction(&mut self, instruction: Instruction<'a>) -> &mut Self {
        self.instructions.push(instruction);
        self
    }
}

impl Display for Section<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{label}:", label = self.label.0)?;
        self.instructions
            .iter()
            .try_fold((), |_, instruction| writeln!(f, "  {instruction}"))
    }
}

struct AsmFunction<'a> {
    name: &'a str,
    move_args: Vec<Instruction<'a>>,
    sections: Vec<Section<'a>>,
}

impl<'a> AsmFunction<'a> {
    fn new(name: &'a str, move_args: impl IntoIterator<Item = Instruction<'a>>) -> Self {
        Self {
            name,
            move_args: move_args.into_iter().collect(),
            sections: Vec::new(),
        }
    }

    fn add_section(&mut self, section: Section<'a>) {
        self.sections.push(section);
    }
}

impl Display for AsmFunction<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, ".text")?;
        writeln!(f, ".globl {func}", func = self.name)?;
        writeln!(f, ".type {func}, @function", func = self.name)?;
        writeln!(f, "{func}:", func = self.name)?;
        self.move_args
            .iter()
            .try_fold((), |_, instruction| writeln!(f, "  {instruction}"))?;
        self.sections
            .iter()
            .try_fold((), |_, section| writeln!(f, "{section}"))
    }
}

#[derive(Default)]
pub struct AsmArchive<'a> {
    ro_data: HashMap<&'a str, StringDef<'a>>,
    bss_data: HashMap<&'a str, Global>,
    functions: Vec<AsmFunction<'a>>,
}

impl<'a> AsmArchive<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    fn get_string(&self, s: &'a str) -> StringRef {
        self.ro_data.get(s).copied().unwrap().get_ref()
    }

    /// Adds strings as read only data. The strings should not include the two double quotes!!
    fn add_string(&mut self, s: &'a str) -> StringRef {
        let len = self.ro_data.len();
        self.ro_data
            .entry(s)
            .or_insert_with(|| {
                let id = StringRef(len as u16);
                StringDef {
                    string: s,
                    id: id.0,
                }
            })
            .get_ref()
    }

    fn add_global(&mut self, name: &'a str, size: usize) {
        let id = self.bss_data.len() as u16;
        self.bss_data.insert(name, Global { id, size });
    }

    fn get_global(&self, name: &'a str) -> MemVar {
        MemVar::Global(GlobalRef(self.bss_data.get(name).unwrap().id))
    }

    fn add_function(&mut self, function: AsmFunction<'a>) {
        self.functions.push(function);
    }
}

impl Display for AsmArchive<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, ".text")?;
        writeln!(f, ".section .rodata")?;
        self.ro_data
            .values()
            .try_fold((), |_, string_def| writeln!(f, "{string_def}"))?;

        writeln!(f, "\n.text")?;
        writeln!(f, ".section .bss")?;
        self.bss_data
            .iter()
            .try_fold((), |_, (_, global_def)| writeln!(f, "{global_def}"))?;

        self.functions
            .iter()
            .try_fold((), |_, function| writeln!(f, "\n{function}"))
    }
}

use crate::ir::{Dest as IRDest, Instruction as IRInstruction, Source as IRSource};

impl<'a> IRSource<'a> {
    fn move_to_reg(
        &self,
        dest: MacReg,
        tmp: MacReg,
        mut symbol_to_mem: impl FnMut(Symbol<'a>) -> MemVar,
        mut reg_to_stack: impl FnMut(u32) -> StackOffset,
        section: &mut Section<'a>,
    ) -> bool {
        use Instruction::*;
        let mut reg_to_stack = |reg: u32| MemVar::Stack(reg_to_stack(reg));
        match self {
            IRSource::Reg(reg) => {
                section.add_instruction(Instruction::Mov(SourceDest::from((
                    reg_to_stack(reg.num()),
                    dest,
                ))));
                false
            }
            IRSource::Sc(reg) => {
                section.add_instruction(Instruction::Mov(SourceDest::from((
                    reg_to_stack(reg.num()),
                    dest,
                ))));
                false
            }
            &IRSource::Offset(sym, reg) => {
                section
                    .add_instruction(Leaq(symbol_to_mem(sym), dest))
                    .add_instruction(Mov(SourceDest::from((reg_to_stack(reg.num()), tmp))))
                    .add_instruction(Mov(SourceDest::IndReg {
                        source: Index {
                            base: dest,
                            offset: tmp,
                        },
                        dest,
                    }));
                true
            }
            &IRSource::Symbol(sym) => {
                section.add_instruction(Instruction::Mov(SourceDest::from((
                    symbol_to_mem(sym),
                    dest,
                ))));
                false
            }
            &IRSource::Immediate(imm) => {
                section.add_instruction(Instruction::Mov(SourceDest::from((i64::from(imm), dest))));
                false
            }
        }
    }
}

impl<'a> IRDest<'a> {
    fn assign(
        self,
        source: MacReg,
        tmp1: MacReg,
        tmp2: MacReg,
        mut symbol_to_mem: impl FnMut(Symbol<'a>) -> MemVar,
        mut reg_to_stack: impl FnMut(u32) -> StackOffset,
        section: &mut Section<'a>,
    ) -> bool {
        use Instruction::*;
        let mut reg_to_stack = |reg: u32| MemVar::Stack(reg_to_stack(reg));
        match self {
            IRDest::Reg(reg) => {
                section.add_instruction(Instruction::Mov(SourceDest::from((
                    source,
                    reg_to_stack(reg.num()),
                ))));
                false
            }
            IRDest::Sc(reg) => {
                section.add_instruction(Instruction::Mov(SourceDest::from((
                    source,
                    reg_to_stack(reg.num()),
                ))));
                false
            }
            IRDest::Offset(sym, reg) => {
                section
                    .add_instruction(Leaq(symbol_to_mem(sym), tmp1))
                    .add_instruction(Mov(SourceDest::from((reg_to_stack(reg.num()), tmp2))))
                    .add_instruction(Mov(SourceDest::RegInd {
                        dest: Index {
                            base: tmp1,
                            offset: tmp2,
                        },
                        source,
                    }));
                true
            }
            IRDest::Symbol(sym) => {
                section.add_instruction(Instruction::Mov(SourceDest::from((
                    source,
                    symbol_to_mem(sym),
                ))));
                false
            }
        }
    }
}

impl<'a> IRInstruction<'a> {
    fn to_macinstructions(
        &self,
        section: &mut Section<'a>,
        mut symbol_to_mem: impl FnMut(Symbol<'a>) -> MemVar,
        mut reg_to_stack: impl FnMut(u32) -> StackOffset,
        mut get_string: impl FnMut(&'a str) -> StringRef,
    ) {
        use Instruction::*;
        match &self {
            Self::AllocArray { .. } | Self::AllocScalar { .. } => {}
            Self::InitSymbol { name } => {
                section
                    .add_instruction(Instruction::Mov(SourceDest::from((0, R10))))
                    .add_instruction(Instruction::Mov(SourceDest::from((
                        R10,
                        symbol_to_mem(*name),
                    ))));
            }
            Self::InitArray { name, size } => {
                section
                    .add_instruction(Instruction::Leaq(symbol_to_mem(*name), ARG_REGISTERS[0]))
                    .add_instruction(Instruction::Mov(SourceDest::from((0, ARG_REGISTERS[1]))))
                    .add_instruction(Instruction::Mov(SourceDest::from((
                        size.get() as i64 * 8,
                        ARG_REGISTERS[2],
                    ))))
                    .add_instruction(Instruction::Mov(SourceDest::from((0, Rax))))
                    .add_instruction(Instruction::Call("memset"));
            }
            Self::Move { dest, source } => {
                source.move_to_reg(R10, R11, &mut symbol_to_mem, &mut reg_to_stack, section);
                dest.assign(
                    R10,
                    R11,
                    Rax,
                    &mut symbol_to_mem,
                    &mut reg_to_stack,
                    section,
                );
            }
            &Self::Return { value } => {
                value.move_to_reg(Rax, R10, &mut symbol_to_mem, &mut reg_to_stack, section);
                section.add_instruction(Leave).add_instruction(Ret);
            }
            Self::VoidReturn => {
                section.add_instruction(Leave).add_instruction(Ret);
            }
            &Self::Unary {
                dest,
                source,
                op: Unary::Not,
            } => {
                source.move_to_reg(R10, R11, &mut symbol_to_mem, &mut reg_to_stack, section);
                section.add_instruction(Xor(SourceDest::from((1, R10))));
                dest.assign(
                    R10,
                    R11,
                    Rax,
                    &mut symbol_to_mem,
                    &mut reg_to_stack,
                    section,
                );
            }
            &Self::Unary {
                dest,
                source,
                op: Unary::Neg,
            } => {
                source.move_to_reg(R10, R11, &mut symbol_to_mem, &mut reg_to_stack, section);
                section.add_instruction(Neg(R10));
                dest.assign(
                    R10,
                    R11,
                    Rax,
                    &mut symbol_to_mem,
                    &mut reg_to_stack,
                    section,
                );
            }
            &IRInstruction::Select {
                dest,
                cond,
                yes,
                no,
            } => {
                cond.move_to_reg(R10, R11, &mut symbol_to_mem, &mut reg_to_stack, section);
                section
                    .add_instruction(Mov(SourceDest::from((1, R11))))
                    .add_instruction(Cmp(R10, R11));
                yes.move_to_reg(R10, Rax, &mut symbol_to_mem, &mut reg_to_stack, section);
                no.move_to_reg(R11, Rax, &mut symbol_to_mem, &mut reg_to_stack, section);
                section.add_instruction(Cmov(SourceDest::from((R10, R11))));
                dest.assign(
                    R11,
                    R10,
                    Rax,
                    &mut symbol_to_mem,
                    &mut reg_to_stack,
                    section,
                );
            }
            &Self::Op2 { dest, lhs, rhs, op } => {
                lhs.move_to_reg(Rax, R10, &mut symbol_to_mem, &mut reg_to_stack, section);
                rhs.move_to_reg(R11, R10, &mut symbol_to_mem, &mut reg_to_stack, section);
                let lhs = Rax;
                let rhs = R11;
                if let Op::Mul = op {
                    section
                        .add_instruction(Mov(SourceDest::from((lhs, Rax))))
                        .add_instruction(IMul(rhs.into()));
                    dest.assign(
                        Rax,
                        R10,
                        R11,
                        &mut symbol_to_mem,
                        &mut reg_to_stack,
                        section,
                    );
                } else if let Op::Div = op {
                    section
                        .add_instruction(Cqto)
                        .add_instruction(IDiv(rhs.into()));
                    dest.assign(
                        Rax,
                        R10,
                        R11,
                        &mut symbol_to_mem,
                        &mut reg_to_stack,
                        section,
                    );
                } else if let Op::Mod = op {
                    section
                        .add_instruction(Cqto)
                        .add_instruction(IDiv(rhs.into()));
                    dest.assign(
                        Rdx,
                        R10,
                        R11,
                        &mut symbol_to_mem,
                        &mut reg_to_stack,
                        section,
                    );
                } else if op.is_cmp() {
                    section
                        .add_instruction(Cmp(rhs, lhs))
                        .add_instruction(Mov(SourceDest::from((1, lhs))))
                        .add_instruction(Mov(SourceDest::from((0, rhs))))
                        .add_instruction(Instruction::cmove(*op, SourceDest::from((lhs, rhs))));
                    dest.assign(
                        rhs,
                        R10,
                        Rax,
                        &mut symbol_to_mem,
                        &mut reg_to_stack,
                        section,
                    );
                } else if op.is_arith() || op.is_logic() {
                    section
                        // this is not a typo - we want to move the result into the lhs
                        // `sub S, D Subtract source from destination`
                        .add_instruction(Instruction::binop(*op, SourceDest::from((rhs, lhs))));
                    dest.assign(
                        lhs,
                        R10,
                        R11,
                        &mut symbol_to_mem,
                        &mut reg_to_stack,
                        section,
                    );
                }
            }
            Self::Exit(code) => {
                section
                    .add_instruction(Mov(SourceDest::from((*code as i64, ARG_REGISTERS[0]))))
                    .add_instruction(Call("exit"));
            }
            &Self::ExternCall { dest, symbol, args } => {
                args.iter()
                    .skip(ARG_REGISTERS.len())
                    .rev()
                    .for_each(|arg| match arg {
                        &IRExternArg::Source(s) => {
                            s.move_to_reg(R10, R11, &mut symbol_to_mem, &mut reg_to_stack, section);
                            section.add_instruction(Push(R10));
                        }
                        IRExternArg::String(s) => {
                            section
                                .add_instruction(Leaq(get_string(s).into(), Rax))
                                .add_instruction(Push(Rax));
                        }
                    });
                args.iter()
                    .zip(ARG_REGISTERS.iter().copied())
                    .for_each(|(arg, reg)| match arg {
                        &IRExternArg::Source(s) => {
                            s.move_to_reg(reg, R11, &mut symbol_to_mem, &mut reg_to_stack, section);
                        }
                        IRExternArg::String(s) => {
                            section.add_instruction(Leaq(MemVar::ReadOnly(get_string(s)), reg));
                        }
                    });
                section
                    .add_instruction(Mov(SourceDest::from((0, Rax))))
                    .add_instruction(Call(symbol));
                dest.assign(
                    Rax,
                    R10,
                    R11,
                    &mut symbol_to_mem,
                    &mut reg_to_stack,
                    section,
                );
            }
            &Self::Call { dest, symbol, args } => {
                args.iter()
                    .skip(ARG_REGISTERS.len())
                    .rev()
                    .for_each(|&arg| {
                        IRSource::from(arg).move_to_reg(
                            R10,
                            R11,
                            &mut symbol_to_mem,
                            &mut reg_to_stack,
                            section,
                        );
                        section.add_instruction(Push(R10));
                    });
                args.iter()
                    .copied()
                    .zip(ARG_REGISTERS.iter().copied())
                    .for_each(|(arg, reg)| {
                        IRSource::from(arg).move_to_reg(
                            reg,
                            Rax,
                            &mut symbol_to_mem,
                            &mut reg_to_stack,
                            section,
                        );
                    });
                section.add_instruction(Call(symbol));
                dest.assign(
                    Rax,
                    R10,
                    R11,
                    &mut symbol_to_mem,
                    &mut reg_to_stack,
                    section,
                );
            }
            Self::VoidCall { symbol, args } => {
                args.iter()
                    .skip(ARG_REGISTERS.len())
                    .rev()
                    .for_each(|&arg| {
                        IRSource::from(arg).move_to_reg(
                            R10,
                            R11,
                            &mut symbol_to_mem,
                            &mut reg_to_stack,
                            section,
                        );
                        section.add_instruction(Push(R10));
                    });
                args.iter()
                    .copied()
                    .zip(ARG_REGISTERS.iter().copied())
                    .for_each(|(arg, reg)| {
                        IRSource::from(arg).move_to_reg(
                            reg,
                            R10,
                            &mut symbol_to_mem,
                            &mut reg_to_stack,
                            section,
                        );
                    });
                section.add_instruction(Call(symbol));
            }
        }
    }
}

impl<'b> Function<'b> {
    fn codegen(&self, archive: &mut AsmArchive<'b>) {
        use Instruction::*;

        self.graph()
            .psudo_inorder()
            .flat_map(|node| {
                node.instructions()
                    .iter()
                    .filter_map(|instruction| match instruction {
                        IRInstruction::ExternCall { args, .. } => {
                            Some(args.iter().filter_map(|arg| match arg {
                                IRExternArg::String(s) => Some(*s),
                                _ => None,
                            }))
                        }
                        _ => None,
                    })
                    .flatten()
            })
            .for_each(|cstr| {
                archive.add_string(cstr);
            });

        let mut symbols_to_offset = self
            .args()
            .iter()
            .take(ARG_REGISTERS.len())
            .enumerate()
            .map(|(i, arg)| (*arg, MemVar::Stack(StackOffset((i + 1) as i32 * -8))))
            .collect::<HashMap<_, _>>();
        symbols_to_offset.extend(
            self.args()
                .iter()
                .skip(ARG_REGISTERS.len())
                .enumerate()
                .map(|(i, arg)| (*arg, MemVar::Stack(StackOffset(16 + i as i32 * 8)))),
        );

        let mut stack_top = symbols_to_offset.len() as i32 * -8;

        symbols_to_offset.extend(self.graph().dfs().flat_map(|node| {
            node.instructions()
                .iter()
                .filter_map(|instruction| match *instruction {
                    IRInstruction::AllocScalar { name, .. } => Some((name, {
                        stack_top -= 8;
                        MemVar::Stack(StackOffset(stack_top))
                    })),
                    IRInstruction::AllocArray { name, size } => Some((name, {
                        stack_top -= (size.get() * 8) as i32;
                        MemVar::Stack(StackOffset(stack_top))
                    })),
                    _ => None,
                })
                .collect::<Vec<_>>()
        }));

        let reg_table = move |reg: u32| StackOffset((reg as i32 + 1) * -8 + stack_top);
        let sym_table = |sym| {
            symbols_to_offset
                .get(&sym)
                .copied()
                .unwrap_or_else(|| archive.get_global(sym.name()))
        };

        archive.add_function(
            self.graph()
                .psudo_inorder()
                .map(|node| {
                    let mut section = Section::new(NonFunctionLabel(node.id()));
                    node.instructions().iter().for_each(|instruction| {
                        instruction.to_macinstructions(&mut section, sym_table, reg_table, |str| {
                            archive.get_string(str)
                        })
                    });

                    match node.neighbours() {
                        None => {}
                        Some(NeighbouringNodes::Unconditional { next }) => {
                            section.add_instruction(Jmp(NonFunctionLabel(next.id())));
                        }
                        Some(NeighbouringNodes::Conditional { cond, yes, no }) => {
                            IRSource::from(cond).move_to_reg(
                                R10,
                                R11,
                                sym_table,
                                reg_table,
                                &mut section,
                            );
                            section
                                .add_instruction(Instruction::Mov(SourceDest::from((1, R11))))
                                .add_instruction(Instruction::Test(R10, R11))
                                .add_instruction(Jz(NonFunctionLabel(no.id())))
                                .add_instruction(Jmp(NonFunctionLabel(yes.id())));
                        }
                    };
                    section
                })
                .fold(
                    AsmFunction::new(
                        self.name(),
                        std::iter::once(Instruction::Enter(
                            stack_top.unsigned_abs() as usize + 8 * self.allocated_regs() as usize,
                        ))
                        .chain(self.move_args()),
                    ),
                    |mut function, section| {
                        function.add_section(section);
                        function
                    },
                ),
        );
    }

    fn move_args<'a>(&self) -> impl Iterator<Item = Instruction<'a>> + '_ {
        self.args()
            .iter()
            .enumerate()
            .zip(ARG_REGISTERS.iter())
            .map(|((i, _), &reg)| {
                Instruction::Mov(SourceDest::RegMem {
                    source: reg,
                    dest: MemVar::Stack(StackOffset((i + 1) as i32 * -8)),
                })
            })
    }
}

impl<'a> Program<'a> {
    pub fn codegen(&self) -> AsmArchive<'a> {
        let mut archive = AsmArchive::new();
        self.globals().iter().for_each(|(name, size)| {
            archive.add_global(name, *size);
        });
        self.functions()
            .iter()
            .for_each(|function| function.codegen(&mut archive));
        archive
    }
}
