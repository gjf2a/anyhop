// Very basic TSP implementation as a planning domain.
// This is primarily to ensure that different search strategies are exploring the search space
// in the correct ordering.

use crate::{Operator, Method, MethodResult, Goal, Task};
use locations::LocationGraph;

#[derive(Clone,PartialOrd,Ord,PartialEq,Eq,Debug)]
pub struct TSPState {
    path: Vec<usize>,
    num_cities: usize
}

impl TSPState {
    pub fn new(goal: &TSPGoal) -> Self {
        TSPState {path: Vec::new(), num_cities: goal.num_cities()}
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct TSPMove {
    end: usize
}

impl TSPMove {
    pub fn new(end: usize) -> Self {TSPMove {end}}
}

impl Operator for TSPMove {
    type S = TSPState;
    type C = u64;
    type G = TSPGoal;

    fn cost(&self, state: &Self::S, goal: &Self::G) -> Self::C {
        state.path.last().map_or(0, |start| goal.map.get(*start, self.end).unwrap())
    }

    fn zero_cost() -> Self::C {
        0
    }

    fn attempt_update(&self, state: &mut Self::S) -> bool {
        if state.path.contains(&self.end) || self.end >= state.num_cities {
            false
        } else {
            state.path.push(self.end);
            true
        }
    }
}

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct TSPGoal {
    map: LocationGraph<usize,u64>
}

impl TSPGoal {
    pub fn new(distances: Vec<(usize,usize,u64)>) -> Self {
        TSPGoal {map: LocationGraph::new(distances)}
    }

    pub fn num_cities(&self) -> usize {
        self.map.len()
    }
}

impl Goal for TSPGoal {
    type O = TSPMove;
    type M = TSPMethod;
    type S = TSPState;
    type C = u64;

    fn starting_tasks(&self) -> Vec<Task<Self::O, Self::M>> {
        vec![Task::Method(TSPMethod {})]
    }

    fn accepts(&self, state: &Self::S) -> bool {
        self.map.all_locations().iter().all(|loc| state.path.contains(loc))
    }

    fn distance_from(&self, state: &Self::S) -> Self::C {
        (self.map.len() - state.path.len()) as u64
    }
}

#[derive(Copy,Clone,PartialOrd,Ord,PartialEq,Eq,Debug)]
pub struct TSPMethod {}

impl Method for TSPMethod {
    type S = TSPState;
    type G = TSPGoal;
    type O = TSPMove;

    fn apply(&self, state: &Self::S, goal: &Self::G) -> MethodResult<Self::O, Self> {
        MethodResult::TaskLists(goal.map.all_locations().iter()
            .filter(|loc| !state.path.contains(loc))
            .map(|loc| vec![Task::Operator(TSPMove::new(*loc)), Task::Method(TSPMethod {})])
            .collect())
    }
}

