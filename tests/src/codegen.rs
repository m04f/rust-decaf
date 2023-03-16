use super::*;

#[test]
fn dcfadd() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = add(*x, *y);
            assert_eq!(res, x.wrapping_add(*y), "{x} + {y} = {res}");
        })
    });
}

#[test]
fn dcfsub() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = sub(*x, *y);
            assert_eq!(res, x.wrapping_sub(*y), "{x} - {y} = {res}");
        })
    });
}

#[test]
fn dcfmul() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = mul(*x, *y);
            assert_eq!(res, x.wrapping_mul(*y), "{x} * {y} = {res}");
        })
    });
}

#[test]
fn dcfdiv() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = div(*x, *y);
            assert_eq!(res, x.wrapping_div(*y), "{x} / {y} = {res}");
        })
    });
}

#[test]
fn dcfrem() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = rem(*x, *y);
            assert_eq!(res, x.wrapping_rem(*y), "{x} % {y} = {res}");
        })
    });
}

#[test]
fn dcfless() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = less(*x, *y);
            assert_eq!(res, *x < *y, "{x} < {y} = {res}");
        })
    });
}

#[test]
fn dcflesseq() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = lesseq(*x, *y);
            assert_eq!(res, *x <= *y, "{x} <= {y} = {res}");
        })
    });
}

#[test]
fn dcfgreater() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = greater(*x, *y);
            assert_eq!(res, *x > *y, "{x} > {y} = {res}");
        })
    });
}

#[test]
fn dcfgreatereq() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = greatereq(*x, *y);
            assert_eq!(res, *x >= *y, "{x} >= {y} = {res}");
        })
    });
}

#[test]
fn dcfequal() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = equal(*x, *y);
            assert_eq!(res, *x == *y, "{x} == {y} = {res}");
        })
    });
}

#[test]
fn dcfnotequal() {
    let a = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MIN, i64::MAX];
    let b = [0, 1, 2, 3, i64::MAX, i64::MIN, i64::MAX, i64::MIN];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = notequal(*x, *y);
            assert_eq!(res, *x != *y, "{x} != {y} = {res}");
        })
    });
}

#[test]
fn dcfand() {
    let a = [true, false];
    let b = [true, false];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = and(*x, *y);
            assert_eq!(res, *x && *y, "{x} && {y} = {res}");
        })
    });
}

#[test]
fn dcfor() {
    let a = [true, false];
    let b = [true, false];
    a.iter().for_each(|x| {
        b.iter().for_each(|y| unsafe {
            let res = or(*x, *y);
            assert_eq!(res, *x || *y, "{x} || {y} = {res}");
        })
    });
}

#[test]
fn dcfnot() {
    let a = [true, false];
    a.iter().for_each(|x| unsafe {
        let res = not(*x);
        assert_eq!(res, !*x, "!{x} = {res}");
    });
}
