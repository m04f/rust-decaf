use crate::log::*;

pub trait CCError {
    fn msgs(self) -> Vec<(String, (usize, usize))>;
}

pub fn ewrite(
    stream: &mut dyn std::io::Write,
    input_file: impl AsRef<str>,
    err: impl CCError,
) -> std::io::Result<()> {
    err.msgs()
        .iter()
        .try_for_each(|m| writeln!(stream, "{}", format_error(input_file.as_ref(), m.1, &m.0)))
}
