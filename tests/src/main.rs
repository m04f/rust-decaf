#[cfg(test)]
extern "C" {
    fn add(a: i64, b: i64) -> i64;
    fn sub(a: i64, b: i64) -> i64;
    fn mul(a: i64, b: i64) -> i64;
    fn div(a: i64, b: i64) -> i64;
    fn rem(a: i64, b: i64) -> i64;

    fn less(a: i64, b: i64) -> bool;
    fn lesseq(a: i64, b: i64) -> bool;
    fn greater(a: i64, b: i64) -> bool;
    fn greatereq(a: i64, b: i64) -> bool;
    fn equal(a: i64, b: i64) -> bool;
    fn notequal(a: i64, b: i64) -> bool;

    fn and(a: bool, b: bool) -> bool;
    fn or(a: bool, b: bool) -> bool;
    fn not(a: bool) -> bool;
}

#[cfg(test)]
mod codegen;
#[cfg(test)]
mod codegen_proptest;

#[cfg(test)]
mod given_tests;

fn main() {}
