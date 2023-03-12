use crate::{
    hir::{HIRRoot, HIRVar},
    ir::{Cfg, Dot},
};

pub enum Type {
    Int,
    Bool,
}

pub enum Global {
    Scalar {
        name: String,
        r#type: Type,
    },
    Array {
        name: String,
        r#type: Type,
        size: usize,
    },
}

pub struct Program {
    globals: Vec<Global>,
    functions: Vec<Function>,
}

pub struct Function {
    graph: Cfg,
    name: String,
    args: Vec<String>,
}

impl AsRef<Cfg> for Function {
    fn as_ref(&self) -> &Cfg {
        self.graph()
    }
}

impl AsMut<Cfg> for Function {
    fn as_mut(&mut self) -> &mut Cfg {
        self.graph_mut()
    }
}

impl Function {
    pub fn new(name: impl ToString, graph: Cfg, args: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            args,
            graph,
        }
    }

    pub fn args(&self) -> &[String] {
        &self.args
    }

    pub fn to_dot(&self) -> Dot {
        self.as_ref().to_dot(Some(self.name()))
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn graph(&self) -> &Cfg {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut Cfg {
        &mut self.graph
    }
}

impl Program {
    /// creates a new program from globals and functions.
    ///
    /// this allows programs without main function. (well they are not programs now.)
    pub fn new(globals: Vec<Global>, functions: Vec<Function>) -> Self {
        Self { globals, functions }
    }

    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    pub fn globals(&self) -> &[Global] {
        &self.globals
    }
}


