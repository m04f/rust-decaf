use super::*;
use crate::ExitStatus;
use seq_macro::seq;

/// a macro to generate tests.
macro_rules! test_illegal {
    ($name:ident, $num:literal, $hidden:literal) => {
        #[test]
        fn $name() {
            let test = concat!(
                "decaf-tests/semantics",
                $hidden,
                "/illegal/illegal-",
                stringify!($num),
                ".dcf"
            );
            assert_eq!(
                Semantics::run(&mut std::io::sink(), &mut std::io::sink(), test.to_string()),
                ExitStatus::Fail
            )
        }
    };
    ($name:ident, $num:literal) => {
        test_illegal!($name, $num, "");
    };
}

seq!(N in 01..=26 {
    test_illegal!(illegal_~N, N);
});
