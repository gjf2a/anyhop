use std::collections::BTreeSet;

#[derive(Clone)]
pub struct BlockState<B> {
    stacks: BTreeSet<(B,B)>,
    table: BTreeSet<B>,
    clear: BTreeSet<B>,
    holding: Option<B>
}
/*
impl BlockState<B> {
    pub fn new(blocks: &Vec<B>) -> Self {
        let mut state = BlockState {stacks: BTreeSet::new(), table: BTreeSet::new(), clear: BTreeSet::new(), holding: None};
        for block in blocks {
            state.table.insert(block);
            state.clear.insert(block);
        }
        state
    }

    pub fn pick_up(&mut self, block: B) -> bool {
        if self.holding == None && self.table.contains(block) {
            self.holding = Some(block);
            self.table.remove(block);
            true
        } else {
            false
        }
    }
}

pub fn pick_up<B>(args: &Vec<B>) -> BlockState<B> {

}
*/