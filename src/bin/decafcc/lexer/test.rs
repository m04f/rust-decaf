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
    test!(char1);
    test!(char2);
    test!(char3);
    test!(char4);
    test!(char5);
    test!(char6);
    test!(char7);
    test!(char8);
    test!(char9);
    test!(hexlit1);
    test!(hexlit2);
    test!(id1);
    test!(id2);
    test!(id3);
    test!(number1);
    test!(number2);
    test!(number3);
    test!(op1);
    test!(op2);
    test!(op3);
    test!(string1);
    test!(string2);
    test!(string3);
    test!(tokens1);
    test!(tokens2);
    test!(tokens3);
    test!(tokens4);
    test!(ws1);
    test!(ws2);
}

mod hidden {
    test_hidden!(char10);
    test_hidden!(char11);
    test_hidden!(char12);
    test_hidden!(char13);
    test_hidden!(char14);
    test_hidden!(char15);
    test_hidden!(hexlit4);
    test_hidden!(id4);
    test_hidden!(id5);
    test_hidden!(literal1);
    test_hidden!(literal2);
    test_hidden!(literal3);
    test_hidden!(literal4);
    test_hidden!(literal5);
    test_hidden!(literal6);
    test_hidden!(literals);
    test_hidden!(number4);
    test_hidden!(number5);
    test_hidden!(op4);
    test_hidden!(op5);
    test_hidden!(op6);
    test_hidden!(op7);
    test_hidden!(op8);
    test_hidden!(string4);
    test_hidden!(tokens5);
    test_hidden!(tokens6);
    test_hidden!(variants);
}
