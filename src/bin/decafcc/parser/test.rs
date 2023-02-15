mod legal {
    use crate::{parser::Parser, App, ExitStatus};

    macro_rules! test_legal {
        ($name:ident, $num:literal) => {
            #[test]
            fn $name() {
                let test = concat!("decaf-tests/parser/legal/legal-", stringify!($num), ".dcf");
                assert_eq!(
                    Parser::run(&mut std::io::sink(), &mut std::io::sink(), test.to_string()),
                    ExitStatus::Success
                )
            }
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
}
