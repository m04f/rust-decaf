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
}
