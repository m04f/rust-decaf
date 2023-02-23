/// a macro to collect stdout and stderr from the given file
macro_rules! collect {
    ($file:expr, $dir:literal) => {{
        fn bytes_to_string(raw: &[u8]) -> String {
            String::from_utf8(raw.to_vec()).unwrap()
        }

        let bytes = include_bytes!(concat!(
            "../../../../decaf-tests/",
            $dir,
            "/output/",
            stringify!($file),
            ".out"
        ));
        let (stdout, stderr): (Vec<_>, Vec<_>) = bytes
            .split(|&c| c == b'\n')
            .filter(|l| !l.is_empty())
            .partition(|l| l[0].is_ascii_digit());
        let (stdout, stderr) = (
            stdout
                .into_iter()
                .map(|line| bytes_to_string(line))
                .collect::<Vec<_>>(),
            stderr
                .into_iter()
                .map(|line| bytes_to_string(line))
                .collect::<Vec<_>>(),
        );
        (stdout, stderr)
    }};
    ($file:expr) => {{
        collect!($file, "scanner")
    }};
}

/// a macro to run the scanner with the given file and return stdout and stderr
macro_rules! run {
    ($file:expr, $dir:literal) => {{
        use crate::{lexer::Lexer, App};
        fn bytes_to_string(raw: &[u8]) -> String {
            String::from_utf8(raw.to_vec()).unwrap()
        }

        let mut stdout = vec![];
        let mut stderr = vec![];
        let exit_status = Lexer::run(
            &mut stdout,
            &mut stderr,
            concat!("decaf-tests/", $dir, "/input/", stringify!($file), ".dcf").to_string(),
        );
        let stdout = stdout
            .split(|&c| c == b'\n')
            .filter(|l| !l.is_empty())
            .map(|line| bytes_to_string(line))
            .collect::<Vec<_>>();
        let stderr = stderr
            .split(|&c| c == b'\n')
            .filter(|l| !l.is_empty())
            .map(|line| bytes_to_string(line))
            .collect::<Vec<_>>();
        (exit_status, stdout, stderr)
    }};
    ($file:expr) => {{
        run!($file, "scanner")
    }};
}

/// generate a test for the given file
macro_rules! test {
    ($file:ident, $dir:literal) => {
        #[test]
        fn $file() {
            use crate::lexer::ExitStatus;
            let (exp_stdout, exp_stderr) = collect!($file, $dir);
            let (exit_status, stdout, _stderr) = run!($file, $dir);
            let exp_exit_status = match exp_stderr.len() {
                0 => ExitStatus::Success,
                _ => ExitStatus::Fail,
            };
            // when there is an error it is quite hard to reproduce their error messages
            // so we just check the exit status
            if exp_exit_status == ExitStatus::Fail {
                assert_eq!(exit_status, exp_exit_status);
            } else {
                assert_eq!(exit_status, exp_exit_status);
                assert_eq!(stdout, exp_stdout);
            }
        }
    };
    ($file:ident) => {
        test!($file, "scanner");
    };
}

macro_rules! test_hidden {
    ($file:ident) => {
        test!($file, "scanner-hidden");
    };
}

mod public {
    use seq_macro::seq;
    seq!(N in 1..=9 {
        test!(char~N);
    });
    seq!(N in 1..=2 {
        test!(hexlit~N);
    });
    seq!(N in 1..=3 {
        test!(id~N);
        test!(number~N);
        test!(op~N);
        test!(string~N);
    });
    seq!(N in 1..=4 {
        test!(tokens~N);
    });
    seq!(N in 1..=2 {
        test!(ws~N);
    });
}

mod hidden {
    use seq_macro::seq;
    seq!(N in 10..=15 {
        test_hidden!(char~N);
    });
    test_hidden!(hexlit4);
    seq!(N in 4..=5 {
        test_hidden!(id~N);
    });
    seq!(N in 1..=6 {
        test_hidden!(literal~N);
    });
    test_hidden!(literals);
    seq!(N in 4..=5 {
        test_hidden!(number~N);
    });
    seq!(N in 4..=8 {
        test_hidden!(op~N);
    });
    test_hidden!(string4);
    seq!(N in 5..=6 {
        test_hidden!(tokens~N);
    });
    test_hidden!(variants);
}
