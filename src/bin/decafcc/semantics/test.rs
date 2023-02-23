use super::*;
use crate::ExitStatus;
use seq_macro::seq;

/// a macro to generate tests for illegal files.
macro_rules! test_illegal {
    ($name:ident, $num:literal, $hidden:literal, $prefix:literal) => {
        #[test]
        fn $name() {
            let test = concat!(
                "decaf-tests/semantics",
                $hidden,
                "/illegal/",
                $prefix,
                "illegal-",
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
        test_illegal!($name, $num, "", "");
    };
}

macro_rules! test_legal {
    ($name:ident, $num:literal) => {
        #[test]
        fn $name() {
            let test = concat!(
                "decaf-tests/semantics-hidden/legal/hidden-legal-",
                stringify!($num),
                ".dcf"
            );

            assert_eq!(
                Semantics::run(&mut std::io::sink(), &mut std::io::sink(), test.to_string()),
                ExitStatus::Success
            )
        }
    };
}

seq!(N in 01..=26 {
    test_illegal!(illegal_~N, N);
});

seq!(N in 01..=07 {
    test_illegal!(hidden_illegal_rule_01_~N, N, "-hidden", "rule-01-");
});
seq!(N in 01..=06 {
    test_illegal!(hidden_illegal_rule_02_~N, N, "-hidden", "rule-02-");
});
seq!(N in 01..=06 {
    test_illegal!(hidden_illegal_rule_03_~N, N, "-hidden", "rule-03-");
});
seq!(N in 01..=02 {
    test_illegal!(hidden_illegal_rule_04_~N, N, "-hidden", "rule-04-");
});
seq!(N in 01..=06 {
    test_illegal!(hidden_illegal_rule_05_~N, N, "-hidden", "rule-05-");
});
test_illegal!(hidden_illegal_rule_06_01, 01, "-hidden", "rule-06-");
seq!(N in 01..=02 {
    test_illegal!(hidden_illegal_rule_07_~N, N, "-hidden", "rule-07-");
});
test_illegal!(hidden_illegal_rule_08_01, 01, "-hidden", "rule-08-");
seq!(N in 01..=04 {
    test_illegal!(hidden_illegal_rule_09_~N, N, "-hidden", "rule-09-");
});
seq!(N in 01..=02 {
    test_illegal!(hidden_illegal_rule_10_~N, N, "-hidden", "rule-10-");
});
seq!(N in 01..=02 {
    test_illegal!(hidden_illegal_rule_11_~N, N, "-hidden", "rule-11-");
});
seq!(N in 01..=02 {
    test_illegal!(hidden_illegal_rule_12_~N, N, "-hidden", "rule-12-");
});
seq!(N in 01..=02 {
    test_illegal!(hidden_illegal_rule_13_~N, N, "-hidden", "rule-13-");
});
seq!(N in 01..=03 {
    test_illegal!(hidden_illegal_rule_14_~N, N, "-hidden", "rule-14-");
});
seq!(N in 01..=03 {
    test_illegal!(hidden_illegal_rule_15_~N, N, "-hidden", "rule-15-");
});
seq!(N in 01..=04 {
    test_illegal!(hidden_illegal_rule_16_~N, N, "-hidden", "rule-16-");
});
seq!(N in 01..=03 {
    test_illegal!(hidden_illegal_rule_17_~N, N, "-hidden", "rule-17-");
});
seq!(N in 01..=03 {
    test_illegal!(hidden_illegal_rule_18_~N, N, "-hidden", "rule-18-");
});
seq!(N in 01..=08 {
    test_illegal!(hidden_illegal_rule_19_~N, N, "-hidden", "rule-19-");
});
seq!(N in 01..=07 {
    test_illegal!(hidden_illegal_rule_20_~N, N, "-hidden", "rule-20-");
});
seq!(N in 01..=03 {
    test_illegal!(hidden_illegal_rule_21_~N, N, "-hidden", "rule-21-");
});
seq!(N in 01..=02 {
    test_illegal!(hidden_illegal_rule_22_~N, N, "-hidden", "rule-22-");
});

seq!(N in 01..=20 {
    test_legal!(legal_~N, N);
});
