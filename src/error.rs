use core::fmt::Display;

const ANSI_RED: &'static str = "\x1b[31m";
const ANSI_RST: &'static str = "\x1b[0m";

pub trait CCError {
    fn msgs(&self) -> Vec<(String, (usize, usize))>;
    fn to_error(self, file: &str) -> Error<Self>
    where
        Self: Sized,
    {
        Error { file, error: self }
    }
}

pub struct Error<'a, T: CCError> {
    file: &'a str,
    error: T,
}

impl<T: CCError> Display for Error<'_, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.msgs().iter().try_fold((), |_, msg| {
            writeln!(
                f,
                "{}:{}:{}: {}error{}: {}",
                self.file, msg.1 .0, msg.1 .1, ANSI_RED, ANSI_RST, msg.0,
            )
        })
    }
}
