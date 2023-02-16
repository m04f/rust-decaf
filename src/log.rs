use std::io::{self, Write};

pub struct Logger<S: Write>(S);

impl Default for Logger<io::Stderr> {
    fn default() -> Self {
        Self::new()
    }
}

impl Logger<io::Stderr> {
    pub fn new() -> Self {
        Self(io::stderr())
    }
}

impl<S: Write> Logger<S> {
    pub fn new_with_stream(stream: S) -> Self {
        Self(stream)
    }
}

/// enum to represint ansi colors
#[allow(unused)]
enum Color {
    Red,
    Green,
    Yellow,
    Blue,
    Magenta,
    Cyan,
    White,
}

impl Color {
    fn to_ansi(&self) -> &'static str {
        match self {
            Color::Red => "\x1b[31m",
            Color::Green => "\x1b[32m",
            Color::Yellow => "\x1b[33m",
            Color::Blue => "\x1b[34m",
            Color::Magenta => "\x1b[35m",
            Color::Cyan => "\x1b[36m",
            Color::White => "\x1b[37m",
        }
    }

    fn reset(&self) -> &'static str {
        "\x1b[0m"
    }
}

#[allow(unused)]
impl<S: Write> Logger<S> {
    /// prints colored messages to the stream
    fn print_color(&mut self, color: Color, msg: &str) {
        write!(self.0, "{}{}{}", color.to_ansi(), msg, color.reset()).unwrap();
    }

    /// prints without chainging the current color
    fn print(&mut self, msg: &str) {
        write!(self.0, "{}", msg).unwrap();
    }

    /// error messages' format is taken from gcc's output
    pub fn log_error<T: AsRef<str>>(
        &mut self,
        file_name: T,
        (line, column): (usize, usize),
        msg: &str,
    ) {
        write!(self.0, "{}", format_error(file_name, (line, column), msg)).unwrap();
    }

    pub fn log_warning(&mut self, file_name: Option<&str>, (line, column): (u32, u32), msg: &str) {
        if file_name.is_none() {
            write!(
                self.0,
                "{}Warning{}: {}\n{}:{}",
                Color::Yellow.to_ansi(),
                Color::Yellow.reset(),
                msg,
                line,
                column
            )
        } else {
            write!(
                self.0,
                "{}Warning{}: {}\n{}:{}:{}",
                Color::Yellow.to_ansi(),
                Color::Yellow.reset(),
                msg,
                file_name.unwrap(),
                line,
                column
            )
        }
        .unwrap()
    }
}

pub fn format_error<F: AsRef<str>, M: AsRef<str>>(
    file_name: F,
    position: (usize, usize),
    msg: M,
) -> String {
    format!(
        "{}:{}:{}: {}error{}: {}",
        file_name.as_ref(),
        position.0,
        position.1,
        Color::Red.to_ansi(),
        Color::Red.reset(),
        msg.as_ref()
    )
}
