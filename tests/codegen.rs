extern "C" {
    fn add(a: i64, b: i64) -> i64;
    fn sub(a: i64, b: i64) -> i64;
    fn mul(a: i64, b: i64) -> i64;
    fn div(a: i64, b: i64) -> i64;
    fn rem(a: i64, b: i64) -> i64;

    fn less(a: i64, b: i64) -> i64;
    fn lesseq(a: i64, b: i64) -> i64;
    fn greater(a: i64, b: i64) -> i64;
    fn greatereq(a: i64, b: i64) -> i64;
    fn equal(a: i64, b: i64) -> i64;
    fn notequal(a: i64, b: i64) -> i64;
}

mod test {
    use super::*;

    fn dcfadd() {
        let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
        let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
        a.iter().for_each(|x| {
            b.iter().for_each(|y| unsafe {
                let res = add(*x, *y);
                assert_eq!(res, x.wrapping_add(*y));
            })
        });
    }
}
