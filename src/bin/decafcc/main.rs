mod lexer;

pub enum ExitStatus {
    Success,
    Fail,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Lexer,
}

struct Config {
    mode: Option<Mode>,
    input_file: Option<String>,
}

impl Config {
    fn new() -> Self {
        Self {
            mode: None,
            input_file: None,
        }
    }

    fn get_mode<T: AsRef<str>>(mode: T) -> Option<Mode> {
        match mode.as_ref() {
            "scan" => Some(Mode::Lexer),
            "scanner" => Some(Mode::Lexer),
            _ => None,
        }
    }

    fn parse(args: impl Iterator<Item = String>) -> Self {
        fn parse(mut config: Config, mut args: impl Iterator<Item = String>) -> Config {
            let first_arg = args.next();
            if let Some(arg) = first_arg {
                match arg.as_str() {
                    "-t" => {
                        config.mode = Some(Config::get_mode(args.next().unwrap())
                                           .unwrap());
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
    use std::{
        env::args,
        fs,
        io::{self, Read},
    };

    let config = Config::parse(args());
    eprintln!("mode: {}", format!("{:?}", config.mode.unwrap()).to_lowercase());
    let input = {
        let mut buffer = vec![];
        config
            .input_file
            .map(|file| fs::File::open(file).unwrap().read_to_end(&mut buffer))
            .unwrap_or(io::stdin().read_to_end(&mut buffer))
            .unwrap();
        buffer
    };
    match config.mode {
        Some(Mode::Lexer) => {
            lexer::run(&input);
        }
        None => {
            println!("No mode specified");
        }
    }
    {}
}
