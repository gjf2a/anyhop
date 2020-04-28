use super::blocks_operators::*;
use crate::Atom;

pub fn is_done<B:Atom>(b1: B, state: &BlockState<B>, goal: &BlockState<B>) -> bool {
    let pos = state.get_pos(b1);
    pos == goal.get_pos(b1) && match pos {
        BlockPos::On(b2) => is_done(b2, state, goal),
        BlockPos::Table => true
    }
}

pub enum Status {
    Done,
    Inaccessible,
    Table,
    MoveToTable,
    MoveToBlock,
    Waiting
}

impl Status {
    pub fn new<B:Atom>(b: B, state: &BlockState<B>, goal: &BlockState<B>) -> Self {
        if is_done(b, state, goal) {
            Status::Done
        } else if !state.clear(b) {
            Status::Inaccessible
        } else {
            match goal.get_pos(b) {
                BlockPos::Table => Status::MoveToTable,
                BlockPos::On(b2) => if state.clear(b2) {
                    Status::MoveToBlock
                } else {
                    Status::Waiting
                }
            }
        }
    }
}

pub enum BlockMethod<B:Atom> {
    MoveBlocks(BlockState<B>, BlockState<B>),
    MoveOne(BlockState<B>, B, BlockPos<B>),
    Get(BlockState<B>, B),
    Put(BlockState<B>, B, B)
}