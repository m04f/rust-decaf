pub mod peephole {
    use crate::ir::Graph;
    impl Graph {
        /// perfonrs peephole optimization on the graph.
        pub fn peephole(&mut self) {
            // the ordering does not matter here anyways
            let mut dfs = self.dfs_mut();
            while let Some(mut node) = dfs.next() {
                while node
                    .as_mut()
                    .merge_tail_if(|tail| tail.insrtuctions().is_empty())
                {}
            }
        }
    }
}
