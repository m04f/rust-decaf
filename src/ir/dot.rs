use std::{
    fmt::Display,
    io::{self, Write},
    path::Path,
    process::{Command, ExitStatus, Stdio},
};

use crate::ir::{Graph, NeighbouringNodes, Node};

pub struct Dot(String);

impl Display for Dot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Dot {
    /// compiles the given graph. (it currently compiles to svg image no matter what).
    ///
    /// returns the exit status of the `dot` command.
    pub fn compile(&self, out_file: impl AsRef<Path>) -> io::Result<ExitStatus> {
        let mut dot_proc = Command::new("dot")
            .arg("-Tsvg")
            .arg("-o")
            .arg(out_file.as_ref())
            .stdin(Stdio::piped())
            .spawn()?;
        dot_proc
            .stdin
            .as_ref()
            .unwrap()
            .write_all(self.0.as_bytes())?;
        dot_proc.wait()
    }

    /// displays the graphs in the systems image viewer (uses xdg-open).
    ///
    /// `graphs` is a list of compiled graphs to be displayed.
    ///
    /// returns the exit status of the `xdg-open` command.
    pub fn display<P: AsRef<Path>>(graphs: impl IntoIterator<Item = P>) -> io::Result<ExitStatus> {
        let mut proc = Command::new("xdg-open");
        for graph in graphs {
            proc.arg(graph.as_ref());
        }
        proc.spawn()?.wait()
    }
}

impl Node<'_> {
    fn dot_label(&self) -> String {
        self.as_ref()
            .instructions()
            .iter()
            .fold(String::new(), |mut acc, instruction| {
                acc.push_str(&format!("{instruction:?}\\n"));
                acc
            })
    }

    /// returns the **out goining** edges
    fn dot_links(&self) -> String {
        match self.neighbours() {
            None => String::default(),
            Some(NeighbouringNodes::Conditional { yes, no, .. }) => {
                // TODO: print the condition register
                format!("{} -> {{{} {}}}\n", self.id(), yes.id(), no.id())
            }
            Some(NeighbouringNodes::Unconditional { next }) => {
                format!("{} -> {}\n", self.id(), next.id())
            }
        }
    }
}

impl Graph<'_> {
    /// converts the graph to a dot graph.
    pub fn to_dot<T: Display>(&self, name: Option<T>) -> Dot {
        let graph_body = self
            .dfs()
            .map(|node| {
                println!("address: {:?}", node.as_ref() as *const _);
                println!("{:?}", node.as_ref());
                format!(
                    // leaving two spaces for indentation.
                    "  {id} [label=\"{label}\" shape=box]\n  {links}\n",
                    id = node.id(),
                    label = node.dot_label(),
                    links = node.dot_links(),
                )
            })
            .collect::<String>();
        if let Some(name) = name {
            Dot(format!(
                "digraph {name} {{\n{graph_body}\n}}",
                graph_body = graph_body,
            ))
        } else {
            Dot(format!(
                "digraph {{\n{graph_body}\n}}",
                graph_body = graph_body,
            ))
        }
    }
}
