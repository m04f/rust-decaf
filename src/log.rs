use std::io::{self, Write};

pub struct Logger<S: Write>(S);

impl Logger<io::Stderr> {
    pub fn new() -> Self {
        Self(io::stderr())
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
    pub fn log_error(&mut self, file_name: Option<&str>, (line, column): (u32, u32), msg: &str) {
        write!(
            self.0,
            "{}:{}:{}: {}error{}: {}\n",
            file_name.unwrap_or("<stdin>"),
            line,
            column,
            Color::Red.to_ansi(),
            Color::Red.reset(),
            msg,
        )
        .unwrap();
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

pub fn log_error(file_name: Option<&str>, pos: (u32, u32), msg: &str) {
    let mut logger = Logger::new();
    logger.log_error(file_name, pos, msg);
}

pub fn log_warning(file_name: Option<&str>, pos: (u32, u32), msg: &str) {
    let mut logger = Logger::new();
    logger.log_warning(file_name, pos, msg);
}

/// macro to log errors given position and optional file arguemnt
/// if no file is given it defaults to `<stdin>`
/// example:
/// ```
/// #[macro_use]
/// use dcfrs::loge;
/// loge!((1, 2), "Error message");
/// loge!("/dev/stdin", (2, 2), "invalid escape character {}", b'c')
/// ```
#[macro_export]
macro_rules! loge {
    ($pos:expr, $msg:expr) => {
        $crate::log::log_error(None, $pos, $msg);
    };
    ($file:expr, $pos:expr, $fmt:expr, $($arg:tt)*) => {
        $crate::log::log_error(Some($file), $pos, &format!($fmt, $($arg)*));
    };
    ($pos:expr, $fmt:expr, $($arg:tt)*) => {
        $crate::log::log_error(None, $pos, &format!($fmt, $($arg)*));
    };
}
