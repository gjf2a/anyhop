use super::blocks_operators::*;
use crate::{Atom, Method, MethodTag, Task};

pub fn is_done<B:Atom>(b1: B, state: &BlockState<B>, goal: &BlockState<B>) -> bool {
    let pos = state.get_pos(b1);
    pos == goal.get_pos(b1) && match pos {
        BlockPos::On(b2) => is_done(b2, state, goal),
        BlockPos::Table => true
    }
}

#[derive(Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Debug)]
pub enum Status<B:Atom> {
    Done(B),
    Inaccessible(B),
    Table(B),
    Move(B,BlockPos<B>),
    Waiting(B)
}

impl <B:Atom> Status<B> {
    pub fn new(b: B, state: &BlockState<B>, goal: &BlockState<B>) -> Self {
        if is_done(b, state, goal) {
            Status::Done(b)
        } else if !state.clear(b) {
            Status::Inaccessible(b)
        } else {
            match goal.get_pos(b) {
                BlockPos::Table => Status::Move(b, BlockPos::Table),
                BlockPos::On(b2) => if state.clear(b2) {
                    Status::Move(b, BlockPos::On(b2))
                } else {
                    Status::Waiting(b)
                }
            }
        }
    }
}

#[derive(Copy, Clone, PartialOrd, PartialEq, Ord, Eq, Debug)]
pub enum BlockMethod<B:Atom> {
    MoveBlocks,
    MoveOne(B, BlockPos<B>),
    Get(B),
    Put(B, BlockPos<B>)
}

impl <B:Atom> Atom for BlockMethod<B> {}

impl <B:Atom> Method<BlockState<B>, BlockOperator<B>, BlockMethod<B>, BlockMethod<B>> for BlockMethod<B> {
    fn apply(&self, state: &BlockState<B>, goal: &BlockState<B>) -> Vec<Vec<Task<BlockOperator<B>, BlockMethod<B>>>> {
        use BlockMethod::*;
        match self {
            MoveBlocks => move_blocks(state, goal),
            MoveOne(block, pos) => move_one(state, *block, *pos),
            Get(block) => get(state, *block),
            Put(block, pos) => put(state, *block, *pos)
        }
    }
}

fn move_blocks<B:Atom>(state: &BlockState<B>, goal: &BlockState<B>) -> Vec<Vec<Task<BlockOperator<B>, BlockMethod<B>>>> {
    use BlockMethod::*;
    let status: Vec<Status<B>> = state.all_blocks().iter().map(|b| Status::new(*b, state, goal)).collect();
    for stat in status.iter() {
        if let Status::Move(b, pos) = stat {
            return vec![vec![Task::MethodTag(MoveOne(*b, *pos)), Task::MethodTag(MoveBlocks)]]
        }
    }

    status.iter()
        .filter_map(|s| match s {
            Status::Waiting(b) => Some(vec![Task::MethodTag(MoveOne(*b, BlockPos::Table)),Task::MethodTag(MoveBlocks)]),
            _ => None
        })
        .collect()
}

fn move_one<B:Atom>(state: &BlockState<B>, block: B, pos: BlockPos<B>) -> Vec<Vec<Task<BlockOperator<B>, BlockMethod<B>>>> {
    vec![vec![Task::MethodTag(BlockMethod::Get(block)), Task::MethodTag(BlockMethod::Put(block, pos))]]
}

fn get<'a, B:Atom>(state: &BlockState<B>, block: B) -> Vec<Vec<Task<BlockOperator<B>, BlockMethod<B>>>> {
    if state.clear(block) {
        match state.get_pos(block) {
            BlockPos::Table => vec![vec![Task::Operator(BlockOperator::PickUp(block))]],
            BlockPos::On(block2) => vec![vec![Task::Operator(BlockOperator::Unstack(block, block2))]]
        }
    } else {
        vec![]
    }
}

fn put<'a, B:Atom>(state: &BlockState<B>, block: B, pos: BlockPos<B>) -> Vec<Vec<Task<BlockOperator<B>, BlockMethod<B>>>> {
    if let Some(b) = state.get_holding() {
        match pos {
            BlockPos::Table => vec![vec![Task::Operator(BlockOperator::PutDown(b))]],
            BlockPos::On(b2) => vec![vec![Task::Operator(BlockOperator::Stack(b, b2))]]
        }
    } else {vec![]}
}

impl <B:Atom> MethodTag<BlockState<B>, BlockOperator<B>, BlockMethod<B>, BlockMethod<B>> for BlockMethod<B> {
    fn candidates(&self, state: &BlockState<B>, goal: &BlockState<B>) -> Vec<BlockMethod<B>> {
        vec![*self]
    }
}