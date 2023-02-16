pub trait CCError {
    fn msgs(self) -> Vec<(String, (usize, usize))>;
}
