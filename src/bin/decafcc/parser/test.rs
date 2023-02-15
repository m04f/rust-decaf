use crate::{parser::Parser, App, ExitStatus};

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

test_legal!(legal_01, 01);
test_legal!(legal_02, 02);
test_legal!(legal_03, 03);
test_legal!(legal_04, 04);
test_legal!(legal_05, 05);
test_legal!(legal_06, 06);
test_legal!(legal_07, 07);
test_legal!(legal_08, 08);
test_legal!(legal_09, 09);
test_legal!(legal_10, 10);
test_legal!(legal_11, 11);
test_legal!(legal_12, 12);
test_legal!(legal_13, 13);
test_legal!(legal_14, 14);
test_legal!(legal_15, 15);
test_legal!(legal_16, 16);
test_legal!(legal_17, 17);
test_legal!(legal_18, 18);
test_legal!(legal_19, 19);
test_legal!(legal_20, 20);
test_legal!(legal_21, 21);
test_legal!(legal_22, 22);
test_legal!(legal_23, 23);
test_legal!(legal_24, 24);
test_legal!(legal_25, 25);
test_legal!(legal_26, 26);
test_legal!(legal_27, 27);
test_legal!(legal_28, 28, "-hidden");
test_legal!(legal_29, 29, "-hidden");
test_legal!(legal_30, 30, "-hidden");
test_legal!(legal_31, 31, "-hidden");
test_legal!(legal_32, 32, "-hidden");
test_legal!(legal_33, 33, "-hidden");
test_legal!(legal_34, 34, "-hidden");
test_legal!(legal_35, 35, "-hidden");
test_legal!(legal_36, 36, "-hidden");
test_legal!(legal_37, 37, "-hidden");
test_legal!(legal_38, 38, "-hidden");
test_legal!(legal_39, 39, "-hidden");
test_legal!(legal_40, 40, "-hidden");
test_legal!(legal_41, 41, "-hidden");
test_legal!(legal_42, 42, "-hidden");

test_illegal!(illegal_01, 01);
test_illegal!(illegal_02, 02);
test_illegal!(illegal_03, 03);
test_illegal!(illegal_04, 04);
test_illegal!(illegal_05, 05);
test_illegal!(illegal_06, 06);
test_illegal!(illegal_07, 07);
test_illegal!(illegal_08, 08);
test_illegal!(illegal_09, 09);
test_illegal!(illegal_10, 10);
test_illegal!(illegal_11, 11);
test_illegal!(illegal_12, 12);
test_illegal!(illegal_13, 13);
test_illegal!(illegal_14, 14);
test_illegal!(illegal_15, 15);
test_illegal!(illegal_16, 16);
test_illegal!(illegal_17, 17);
test_illegal!(illegal_18, 18);
test_illegal!(illegal_19, 19);
test_illegal!(illegal_20, 20);
test_illegal!(illegal_21, 21);
test_illegal!(illegal_22, 22);
test_illegal!(illegal_23, 23);
test_illegal!(illegal_24, 24);
test_illegal!(illegal_25, 25);
test_illegal!(illegal_26, 26);
test_illegal!(illegal_27, 27);
test_illegal!(illegal_28, 28);
test_illegal!(illegal_29, 29);
test_illegal!(illegal_30, 30);
test_illegal!(illegal_31, 31);
test_illegal!(illegal_32, 32);
test_illegal!(illegal_33, 33);
test_illegal!(illegal_34, 34);
test_illegal!(illegal_35, 35, "-hidden");
test_illegal!(illegal_36, 36, "-hidden");
test_illegal!(illegal_37, 37, "-hidden");
test_illegal!(illegal_38, 38, "-hidden");
test_illegal!(illegal_39, 39, "-hidden");
test_illegal!(illegal_40, 40, "-hidden");
test_illegal!(illegal_41, 41, "-hidden");
test_illegal!(illegal_42, 42, "-hidden");
test_illegal!(illegal_43, 43, "-hidden");
test_illegal!(illegal_44, 44, "-hidden");
test_illegal!(illegal_45, 45, "-hidden");
test_illegal!(illegal_46, 46, "-hidden");
test_illegal!(illegal_47, 47, "-hidden");
test_illegal!(illegal_48, 48, "-hidden");
test_illegal!(illegal_49, 49, "-hidden");
test_illegal!(illegal_50, 50, "-hidden");
test_illegal!(illegal_51, 51, "-hidden");
test_illegal!(illegal_52, 52, "-hidden");
test_illegal!(illegal_53, 53, "-hidden");
test_illegal!(illegal_54, 54, "-hidden");
test_illegal!(illegal_55, 55, "-hidden");
test_illegal!(illegal_56, 56, "-hidden");
test_illegal!(illegal_57, 57, "-hidden");
test_illegal!(illegal_58, 58, "-hidden");
test_illegal!(illegal_59, 59, "-hidden");
