use super::*;

use proptest::prelude::*;

proptest!(
    #[test]
    fn dcfadd(a: i64, b: i64) {
        unsafe {
            let res = add(a, b);
            assert_eq!(res, a.wrapping_add(b), "{a} + {b} = {res}");
        }
    }

    #[test]
    fn dcfsub(a: i64, b: i64) {
        unsafe {
            let res = sub(a, b);
            assert_eq!(res, a.wrapping_sub(b), "{a} - {b} = {res}");
        }
    }

    #[test]
    fn dcfmul(a: i64, b: i64) {
        unsafe {
            let res = mul(a, b);
            assert_eq!(res, a.wrapping_mul(b), "{a} * {b} = {res}");
        }
    }

    #[test]
    fn dcfdiv(a: i64, mut b: i64) {
        if b == 0 {
            b = 1;
        }
        unsafe {
            let res = div(a, b);
            assert_eq!(res, a.wrapping_div(b), "{a} / {b} = {res}");
        }
    }

    #[test]
    fn dcfrem(a: i64, mut b: i64) {
        if b == 0 {
            b = 1;
        }
        unsafe {
            let res = rem(a, b);
            assert_eq!(res, a.wrapping_rem(b), "{a} % {b} = {res}");
        }
    }
);
