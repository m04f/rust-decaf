use std::io::stderr;

use crate::{lexer::Lexer, parser::Parser, semantics::Semantics};

mod lexer;
mod parser;
mod semantics;

trait App {
    fn run(
        stdout: &mut dyn std::io::Write,
        stderr: &mut dyn std::io::Write,
        input_file: String,
    ) -> ExitStatus;
}

#[derive(Debug, Default, PartialEq, Eq)]
pub enum ExitStatus {
    #[default]
    Success,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Lexer,
    Parser,
    Semantics,
}

struct Config {
    mode: Option<Mode>,
    input_file: Option<String>,
    output_file: Option<String>,
    // stderr: Option<String>,
}

impl Config {
    fn new() -> Self {
        Self {
            mode: None,
            input_file: None,
            output_file: None,
            // stderr: None,
        }
    }

    fn get_mode<T: AsRef<str>>(mode: T) -> Option<Mode> {
        match mode.as_ref() {
            "scan" => Some(Mode::Lexer),
            "scanner" => Some(Mode::Lexer),
            "parser" => Some(Mode::Parser),
            "parse" => Some(Mode::Parser),
            "semantics" => Some(Mode::Semantics),
            "semantic" => Some(Mode::Semantics),
            _ => None,
        }
    }

    fn parse(args: impl Iterator<Item = String>) -> Self {
        fn parse(mut config: Config, mut args: impl Iterator<Item = String>) -> Config {
            let first_arg = args.next();
            if let Some(arg) = first_arg {
                match arg.as_str() {
                    "-t" => {
                        config.mode = Some(Config::get_mode(args.next().unwrap()).unwrap());
                        parse(config, args)
                    }
                    "-o" | "--output" => {
                        config.output_file = Some(args.next().unwrap());
                        parse(config, args)
                    }
                    s if !s.is_empty() => {
                        config.input_file = Some(s.to_string());
                        parse(config, args)
                    }
                    _ => unimplemented!(),
                }
            } else {
                config
            }
        }
        parse(Config::new(), args.skip(1))
    }
}

fn main() {
    use std::{env::args, fs, io};

    let config = Config::parse(args());
    eprintln!(
        "mode: {}",
        format!("{:?}", config.mode.unwrap()).to_lowercase()
    );
    let mut output_stream: Box<dyn io::Write> = config
        .output_file
        .map(|path| {
            Box::new(io::BufWriter::new(fs::File::create(path).unwrap())) as Box<dyn io::Write>
        })
        .unwrap_or(Box::new(stderr()));
    let mut stderr = Box::new(stderr()) as Box<dyn io::Write>;
    match config.mode {
        Some(Mode::Lexer) => Lexer::run(
            &mut output_stream,
            &mut stderr,
            config.input_file.unwrap_or("/dev/stdin".to_string()),
        ),
        Some(Mode::Parser) => Parser::run(
            &mut output_stream,
            &mut stderr,
            config.input_file.unwrap_or("/dev/stdin".to_string()),
        ),
        Some(Mode::Semantics) => Semantics::run(
            &mut output_stream,
            &mut stderr,
            config.input_file.unwrap_or("/dev/stdin".to_string()),
        ),
        None => {
            println!("No mode specified");
            ExitStatus::Fail
        }
    };
}
