use crate::{parser::Parser, App, ExitStatus};
use seq_macro::seq;

macro_rules! test_legal {
    ($name:ident, $num:literal, $hidden:literal) => {
        #[test]
        fn $name() {
            let test = concat!(
                "decaf-tests/parser",
                $hidden,
                "/legal/legal-",
                stringify!($num),
                ".dcf"
            );
            assert_eq!(
                Parser::run(&mut std::io::sink(), &mut std::io::sink(), test.to_string()),
                ExitStatus::Success
            )
        }
    };
    ($name:ident, $num:literal) => {
        test_legal!($name, $num, "");
    };
}

macro_rules! test_illegal {
    ($name:ident, $num:literal, $hidden:literal) => {
        #[test]
        fn $name() {
            let test = concat!(
                "decaf-tests/parser",
                $hidden,
                "/illegal/illegal-",
                stringify!($num),
                ".dcf"
            );
            assert_eq!(
                Parser::run(&mut std::io::sink(), &mut std::io::sink(), test.to_string()),
                ExitStatus::Fail
            )
        }
    };
    ($name:ident, $num:literal) => {
        test_illegal!($name, $num, "");
    };
}

seq!(N in 01..=27 {
    test_legal!(legal_~N, N);
});
seq!(N in 28..=42 {
    test_legal!(legal_~N, N, "-hidden");
});
seq!(N in 01..=34 {
    test_illegal!(illegal_~N, N);
});
seq!(N in 35..=59 {
    test_illegal!(illegal_~N, N, "-hidden");
});
