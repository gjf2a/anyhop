use immutable_map::{TreeSet, TreeMap};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::collections::VecDeque;

pub fn find_first_plan<S:Orderable,O:Atom+Operator<S,A>,M:Atom+Method<S,A,O,M,T>,T:Atom+MethodTag<S,A,O,M,T>,A:Atom>
(state: &S, tasks: &Vec<Task<O,T,A>>, verbose: usize) -> Option<Vec<(O,A)>> {
    let mut p = PlannerStep::new(state, tasks, verbose);
    p.verb(format!("** pyhop, verbose={}: **\n   state = {:?}\n   tasks = {:?}", verbose, state, tasks), 0);
    let mut choices = VecDeque::new();
    while !p.is_complete() {
        let next_options = p.get_next_step();
        for option in next_options {
            choices.push_back(option);
        }
        match choices.pop_back() {
            Some(choice) => {p = choice;},
            None => {
                p.verb(format!("** No plan found **"), 0);
                return None;
            }
        }
    }
    return Some(p.plan);
}

pub trait Orderable : Clone + Debug + Ord + Eq {}

pub trait Atom : Copy + Clone + Debug + Ord + Eq {}

pub trait Operator<S:Clone,A:Atom> {
    fn apply(&self, state: &S, args: A) -> Option<S> {
        let mut updated = state.clone();
        let success = self.attempt_update(&mut updated, args);
        if success {Some(updated)} else {None}
    }

    fn attempt_update(&self, state: &mut S, args: A) -> bool;
}

pub trait Method<S:Clone,A:Atom,O:Atom+Operator<S,A>,M:Atom+Method<S,A,O,M,T>,T:Atom+MethodTag<S,A,O,M,T>> {
    fn apply(&self, args: A) -> Vec<Vec<Task<O, T, A>>>;
}

pub trait MethodTag<S:Clone,A:Atom,O:Atom+Operator<S,A>,M:Atom+Method<S,A,O,M,T>,T:Atom+MethodTag<S,A,O,M,T>> {
    fn candidates(&self) -> Vec<M>;
}

#[derive(Copy,Clone,Debug)]
pub enum Task<O:Atom, T:Atom, A:Atom> {
    Operator(O, A),
    MethodTag(T, A)
}

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct LocationGraph<L: Atom> {
    distances: TreeMap<L,TreeMap<L,usize>>
}

impl <L:Atom> LocationGraph<L> {
    pub fn new(distances: Vec<(L,L,usize)>) -> Self {
        let mut map_graph = LocationGraph {distances: TreeMap::new()};
        for distance in distances.iter() {
            map_graph.add(distance.0, distance.1, distance.2);
        }
        map_graph
    }

    pub fn get(&self, start: L, end: L) -> Option<usize> {
        self.distances.get(&start)
            .and_then(|map| map.get(&end))
            .map(|d| *d)
    }

    pub fn add(&mut self, m1: L, m2: L, distance: usize) {
        self.add_one_way(m1, m2, distance);
        self.add_one_way(m2, m1, distance);
    }

    fn add_one_way(&mut self, start: L, end: L, distance: usize) {
        let updated = self.distances.get(&start)
            .unwrap_or(&TreeMap::new())
            .insert(end, distance);
        self.distances = self.distances.insert(start, updated);
    }
}

pub fn all_or_none<T:Clone>(options: Vec<Option<T>>) -> Option<Vec<T>> {
    if options.iter().all(|op| op.is_some()) {
        Some(options.iter().map(|op| op.as_ref().unwrap().clone()).collect())
    } else {
        None
    }
}

#[derive(Clone)]
struct PlannerStep<S:Orderable,O:Atom+Operator<S,A>,M:Atom+Method<S,A,O,M,T>,T:Atom+MethodTag<S,A,O,M,T>,A:Atom> {
    verbose: usize,
    state: S,
    prev_states: TreeSet<S>,
    tasks: Vec<Task<O,T,A>>,
    plan: Vec<(O,A)>,
    depth: usize,
    _ph: PhantomData<M>
}

impl <S:Orderable,O:Atom+Operator<S,A>,M:Atom+Method<S,A,O,M,T>,T:Atom+MethodTag<S,A,O,M,T>,A:Atom> PlannerStep<S,O,M,T,A> {
    pub fn new(state: &S, tasks: &Vec<Task<O,T,A>>, verbose: usize) -> Self {
        PlannerStep {verbose, state: state.clone(), prev_states: TreeSet::new().insert(state.clone()), tasks: tasks.clone(), plan: vec![], depth: 0, _ph: PhantomData }
    }

    pub fn is_complete(&self) -> bool {
        self.tasks.len() == 0
    }

    pub fn get_next_step(&self) -> Vec<Self> {
        self.verb(format!("depth {} tasks {:?}", self.depth, self.tasks), 1);
        if self.is_complete() {
            self.verb(format!("depth {} returns plan {:?}", self.depth, self.plan), 2);
            vec![self.clone()]
        } else {
            if let Some(task1) = self.tasks.get(0) {
                match task1 {
                    Task::Operator(op, args) => self.apply_operator(*op, *args),
                    Task::MethodTag(tag, args) => self.apply_method(*tag, *args)
                }
            } else {
                self.verb(format!("Depth {} returns failure", self.depth), 2);
                vec![]
            }
        }
    }

    fn apply_operator(&self, operator: O, args: A) -> Vec<Self> {
        if let Some(new_state) = operator.apply(&self.state, args) {
            if self.prev_states.contains(&new_state) {
                self.verb(format!("Cycle; pruning..."), 2);
            } else {
                self.verb(format!("Depth {}; new_state: {:?}", self.depth, new_state), 2);
                return vec![self.operator_planner_step(new_state, operator, args)];
            }
        }
        vec![]
    }

    fn apply_method(&self, tag: T, args: A) -> Vec<Self> {
        let mut planner_steps = Vec::new();
        for candidate in tag.candidates() {
            let subtask_alternatives = candidate.apply(args);
            self.verb(format!("{} alternative subtask lists", subtask_alternatives.len()), 2);
            for subtasks in subtask_alternatives.iter() {
                self.verb(format!("depth {} new tasks: {:?}", self.depth, subtasks), 2);
                planner_steps.push(self.method_planner_step(subtasks));
            }
        }
        planner_steps
    }

    fn operator_planner_step(&self, state: S, operator: O, args: A) -> Self {
        let mut updated_plan = self.plan.clone();
        updated_plan.push((operator, args));
        PlannerStep { verbose: self.verbose, prev_states: self.prev_states.insert(state.clone()), state: state, tasks: self.tasks[1..].to_vec(), plan: updated_plan, depth: self.depth + 1, _ph: PhantomData }
    }

    fn method_planner_step(&self, subtasks: &Vec<Task<O,T,A>>) -> Self {
        let mut updated_tasks = Vec::new();
        subtasks.iter().for_each(|t| updated_tasks.push(*t));
        self.tasks.iter().skip(1).for_each(|t| updated_tasks.push(*t));
        PlannerStep {verbose: self.verbose, prev_states: self.prev_states.clone(), state: self.state.clone(), tasks: updated_tasks, plan: self.plan.clone(), depth: self.depth + 1, _ph: PhantomData}
    }

    fn verb(&self, text: String, level: usize) {
        if self.verbose > level {
            println!("{}", text);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{find_first_plan, Task, Atom, LocationGraph};
    use crate::tests::simple_travel::{TravelState, Args, CityMethodTag, CityOperator};
    use rust_decimal_macros::*;

    mod blocks_operators;
    mod blocks_methods_1;
    mod simple_travel;

    #[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
    enum Location {
        Home, Park, TaxiStand
    }

    impl Atom for Location {}
    impl Atom for char {}

    #[test]
    fn simple_travel_1() {
        use Location::*; use CityOperator::*; use CityMethodTag::*; use Args::*;
        let mut state = TravelState::new(LocationGraph::new(vec![(Home, Park, 8)]), TaxiStand);
        state.add_traveler('M', dec!(20), Home);
        let tasks = vec![Task::MethodTag(Travel, Move('M', Home, Park))];
        let plan = find_first_plan(&state, &tasks, 3).unwrap();
        println!("the plan: {:?}", &plan);
        assert_eq!(plan, vec![(CallTaxi, Traveler('M')), (RideTaxi, Move('M', Home, Park)), (Pay, Traveler('M'))]);
    }
}
