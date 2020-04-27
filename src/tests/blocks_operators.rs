use std::collections::BTreeSet;
use crate::{Atom, Operator};

#[derive(Clone, PartialOrd, PartialEq, Ord, Eq)]
pub struct BlockState<B:Atom> {
    stacks: BTreeSet<(B,B)>,
    table: BTreeSet<B>,
    clear: BTreeSet<B>,
    holding: Option<B>
}

impl <B:Atom> BlockState<B> {
    pub fn new(blocks: &Vec<B>) -> Self {
        let mut state = BlockState {stacks: BTreeSet::new(), table: BTreeSet::new(), clear: BTreeSet::new(), holding: None};
        for block in blocks {
            state.table.insert(*block);
            state.clear.insert(*block);
        }
        state
    }

    pub fn pick_up(&mut self, block: B) -> bool {
        if self.holding == None && self.table.contains(&block) && self.clear.contains(&block) {
            self.holding = Some(block);
            self.table.remove(&block);
            self.clear.remove(&block);
            true
        } else {false}
    }

    pub fn put_down(&mut self, block: B) -> bool {
        if self.holding == Some(block) {
            self.clear.insert(block);
            self.table.insert(block);
            self.holding = None;
            true
        } else {false}
    }

    pub fn unstack(&mut self, a: B, b: B) -> bool {
        if self.holding == None && self.stacks.contains(&(a, b)) && self.clear.contains(&a) {
            self.holding = Some(a);
            self.clear.insert(b);
            self.clear.remove(&a);
            self.stacks.remove(&(a, b));
            true
        } else {false}
    }

    pub fn stack(&mut self, a: B, b: B) -> bool {
        if self.holding == Some(a) && self.clear.contains(&b) {
            self.holding = None;
            self.clear.remove(&b);
            self.clear.insert(a);
            self.stacks.insert((a, b));
            true
        } else {false}
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
enum Args<B:Atom> {
    One(B), Two(B,B)
}

impl <B:Atom> Atom for Args<B> {}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum BlocksOperator {
    PickUp, PutDown, Stack, Unstack
}

impl Atom for BlocksOperator {}

impl <B:Atom> Operator<BlockState<B>,Args<B>> for BlocksOperator {
    fn attempt_update(&self, state: &mut BlockState<B>, args: Args<B>) -> bool {
        use BlocksOperator::*; use Args::*;
        match args {
            One(block) => match self {
                PickUp => state.pick_up(block),
                PutDown => state.put_down(block),
                _ => false
            },
            Two(b1, b2) => match self {
                Stack => state.stack(b1, b2),
                Unstack => state.unstack(b1, b2),
                _ => false
            }
        }
    }
}
