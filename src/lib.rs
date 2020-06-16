#![feature(trait_alias)]

use immutable_map::TreeSet;
use std::fmt::Debug;
use std::collections::VecDeque;
use std::time::Instant;
use num_traits::Num;

pub fn find_first_plan<S,G,O,M>(state: &S, goal: &G, tasks: &Vec<Task<O,M>>, verbose: usize) -> Option<Vec<O>>
    where S:Orderable, G:Goal<M=M,O=O>, O:Operator<S=S>, M:Method<S=S,G=G,O=O> {
    let mut p = PlannerStep::new(state, tasks, verbose);
    p.verb(format!("** anyhop, verbose={}: **\n   state = {:?}\n   tasks = {:?}", verbose, state, tasks), 0);
    let mut choices = VecDeque::new();
    while !p.is_complete() {
        for option in p.get_next_step(goal) {
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
    Some(p.plan)
}

#[derive(Debug,Copy,Clone,Eq,PartialEq)]
pub enum BacktrackPreference {
    MostRecent, LeastRecent
}

#[derive(Debug,Copy,Clone,Eq,PartialEq)]
pub enum BacktrackStrategy {
    Steady(BacktrackPreference),
    Alternate(BacktrackPreference)
}

impl BacktrackStrategy {
    pub fn next(self) -> Self {
        use BacktrackStrategy::*; use BacktrackPreference::*;
        match self {
            Steady(_) => self,
            Alternate(p) => match p {
                LeastRecent => Alternate(MostRecent),
                MostRecent => Alternate(LeastRecent)
            }
        }
    }

    pub fn pref(&self) -> BacktrackPreference {
        match self {
            BacktrackStrategy::Steady(p) => *p,
            BacktrackStrategy::Alternate(p) => *p
        }
    }
}

pub struct AnytimePlannerBuilder<'a,S,G,F>
    where S:Orderable, G:Goal, F: ?Sized {
    state: S, goal: G, time_limit_ms: Option<u128>, strategy: BacktrackStrategy,
    cost_func: &'a F, verbose: usize, apply_cutoff: bool
}

impl <'a,S,G,O,M,C,F> AnytimePlannerBuilder<'a,S,G,F>
    where S:Orderable, O:Operator<S=S>,
          G:Goal<M=M,O=O>, M:Method<S=S,G=G,O=O>,
          C:Cost, F:?Sized + Fn(&Vec<O>) -> C {

    pub fn state_goal_cost(state: &S, goal: &G, cost_func: &'a F) -> Self {
        AnytimePlannerBuilder { state: state.clone(), goal: goal.clone(), time_limit_ms: None,
            strategy: BacktrackStrategy::Steady(BacktrackPreference::MostRecent),
            cost_func, verbose: 0, apply_cutoff: true
        }
    }

    pub fn verbose(&'a mut self, verbose: usize) -> &'a mut Self {
        self.verbose = verbose;
        self
    }

    pub fn time_limit_ms(&'a mut self, time_limit_ms: u128) -> &'a mut Self {
        self.time_limit_ms = Some(time_limit_ms);
        self
    }

    pub fn possible_time_limit_ms(&'a mut self, time_limit_ms: Option<u128>) -> &'a mut Self {
        self.time_limit_ms = time_limit_ms;
        self
    }

    pub fn apply_cutoff(&'a mut self, apply_cutoff: bool) -> &'a mut Self {
        self.apply_cutoff = apply_cutoff;
        self
    }

    pub fn strategy(&'a mut self, strategy: BacktrackStrategy) -> &'a mut Self {
        self.strategy = strategy;
        self
    }

    pub fn construct(&self) -> AnytimePlanner<S,O,M,C> {
        AnytimePlanner::plan(&self.state, &self.goal, self.time_limit_ms, self.strategy, &self.cost_func, self.verbose, self.apply_cutoff)
    }
}

impl <'a,S,G,O,M> AnytimePlannerBuilder<'a,S,G,dyn Fn(&Vec<O>) -> usize>
    where S:Orderable, O:Operator<S=S>,
          G:Goal<M=M,O=O>, M:Method<S=S,G=G,O=O> {

    pub fn state_goal(state: &S, goal: &G) -> Self {
        AnytimePlannerBuilder::state_goal_cost(state, goal, &|v| v.len())
    }
}

pub struct AnytimePlanner<S,O, M,C>
    where S:Orderable, O:Operator, M:Method, C:Cost {
    plans: Vec<Vec<O>>,
    discovery_times: Vec<u128>,
    discovery_prunes: Vec<usize>,
    discovery_prior_plans: Vec<usize>,
    costs: Vec<C>,
    cheapest: Option<C>,
    total_iterations: usize,
    total_pops: usize,
    total_pushes: usize,
    total_pruned: usize,
    start_time: Instant,
    total_time: Option<u128>,
    current_step: PlannerStep<S,O,M>
}

pub fn summary_csv_header() -> String {
    format!("label,first,most_expensive,cheapest,discovery_time,total_time,pruned_prior,num_prior_plans,total_prior_attempts,total_attempts\n")
}

impl <S,O,C,G,M> AnytimePlanner<S,O,M,C>
    where S:Orderable, O:Operator<S=S>,
          C:Cost,
          G:Goal<M=M,O=O>,
          M:Method<S=S,G=G,O=O> {
    fn plan<F:Fn(&Vec<O>) -> C>(state: &S, goal: &G, time_limit_ms: Option<u128>, strategy: BacktrackStrategy, cost_func: &F, verbose: usize, apply_cutoff: bool) -> Self {
        let mut outcome = AnytimePlanner {
            plans: Vec::new(), discovery_times: Vec::new(), cheapest: None, costs: Vec::new(),
            discovery_prior_plans: Vec::new(), discovery_prunes: Vec::new(), total_iterations: 0,
            total_pops: 0, total_pushes: 0, total_pruned: 0, start_time: Instant::now(),
            total_time: None,
            current_step: PlannerStep::new(state, &goal.starting_tasks(), verbose)
        };
        outcome.make_plan(goal, time_limit_ms, strategy, cost_func, apply_cutoff);
        outcome
    }

    fn make_plan<F:Fn(&Vec<O>) -> C>(&mut self, goal: &G, time_limit_ms: Option<u128>, strategy: BacktrackStrategy, cost_func: &F, apply_cutoff: bool) {
        let mut choices = VecDeque::new();
        let mut backtrack = (false, strategy);
        loop {
            self.total_iterations += 1;
            backtrack = if apply_cutoff && self.current_too_expensive(cost_func) {
                let time = self.time_since_start();
                self.total_pruned += 1;
                self.current_step.verb(format!("Plan pruned. Time: {}", time), 1);
                (true, backtrack.1.next())
            } else {
                self.add_choices(goal, backtrack.1, &mut choices, cost_func)
            };
            if choices.is_empty() {
                self.set_total_time();
                self.current_step.verb(format!("** No plans left to be found ({} ms elapsed) **", self.total_time.unwrap()), 0);
                self.current_step.verb(format!("{} attempts, {} found, {} pruned", self.plans.len() + self.total_pruned, self.plans.len(), self.total_pruned), 0);
                break;
            } else if self.time_up(time_limit_ms) {
                self.set_total_time();
                self.current_step.verb(format!("Time's up! {:?} ms elapsed", time_limit_ms), 0);
                break;
            } else {
                self.pick_choice(backtrack, &mut choices);
            }
        }
    }

    fn time_since_start(&self) -> u128 {
        Instant::now().duration_since(self.start_time).as_millis()
    }

    fn set_total_time(&mut self) {
        self.total_time = Some(self.time_since_start());
    }

    fn current_too_expensive<F:Fn(&Vec<O>) -> C>(&self, cost_func: F) -> bool {
        self.cheapest.map_or(false, |bound| cost_func(&self.current_step.plan) >= bound)
    }

    fn time_up(&self, time_limit_ms: Option<u128>) -> bool {
        match time_limit_ms {
            None => false,
            Some(limit) => self.time_since_start() >= limit
        }
    }

    fn add_choices<F:Fn(&Vec<O>) -> C>(&mut self, goal: &G, strategy: BacktrackStrategy, choices: &mut VecDeque<PlannerStep<S,O,M>>, cost_func: &F) -> (bool,BacktrackStrategy) {
        if self.current_step.is_complete() {
            let plan = self.current_step.plan.clone();
            let cost: C = cost_func(&plan);
            self.cheapest = Some(self.cheapest.map_or(cost,|c| if cost < c {cost} else {c}));
            self.add_plan(self.current_step.plan.clone(), cost);
            (true, strategy.next())
        } else {
            for option in self.current_step.get_next_step(goal) {
                choices.push_back(option);
                self.total_pushes += 1;
            }
            (false, strategy)
        }
    }

    fn add_plan(&mut self, plan: Vec<O>, cost: C) {
        self.costs.push(cost);
        let time = self.time_since_start();
        self.discovery_times.push(time);
        self.discovery_prunes.push(self.total_pruned);
        self.discovery_prior_plans.push(self.plans.len());
        self.plans.push(plan);
        self.current_step.verb(format!("Plan found. Cost: {:?}; Time: {}", cost, time), 0);
    }

    fn pick_choice(&mut self, backtrack: (bool, BacktrackStrategy), choices: &mut VecDeque<PlannerStep<S,O,M>>) {
        self.current_step = if backtrack.0 && backtrack.1.pref() == BacktrackPreference::LeastRecent {
            choices.pop_front()
        } else {
            choices.pop_back()
        }.unwrap();
        self.total_pops += 1;
    }

    pub fn index_of_cheapest(&self) -> usize {
        (0..self.costs.len())
            .fold(0, |best, i| if self.costs[i] < self.costs[best] {i} else {best})
    }

    pub fn get_plan(&self, i: usize) -> Vec<O> {
        self.plans[i].clone()
    }

    pub fn get_best_plan(&self) -> Vec<O> {
        self.get_plan(self.index_of_cheapest())
    }

    pub fn get_all_plans(&self) -> Vec<Vec<O>> {
        self.plans.clone()
    }

    fn plan_data(&self, p: usize) -> (u128,usize,usize,usize) {
        (self.discovery_times[p], self.discovery_prunes[p], self.discovery_prior_plans[p],
         self.discovery_prunes[p] + self.discovery_prior_plans[p])
    }

    pub fn summary_csv_row(&self, label: &str) -> String {
        let c = self.index_of_cheapest();
        let (discovery_time, pruned_prior, plans_prior, attempt) = self.plan_data(c);
        format!("{},{:?},{:?},{:?},{},{},{},{},{},{}\n", label, self.costs[0], self.highest_cost(),
                self.lowest_cost(), discovery_time,  self.total_time.unwrap(), pruned_prior,
                plans_prior, attempt, self.total_pruned + self.plans.len())
    }

    pub fn instance_csv(&self) -> String {
        let mut result = String::from("plan_cost,discovery_time,attempt,pruned_prior,plans_prior\n");
        for p in 0..self.index_of_cheapest() {
            let (discovery_time,pruned_prior, plans_prior, attempt) = self.plan_data(p);
            result.push_str(format!("{:?},{},{},{},{}\n", self.costs[p], discovery_time,
                                    attempt, pruned_prior, plans_prior)
                .as_str());
        }
        result
    }

    pub fn lowest_cost(&self) -> C {
        *self.costs.iter().min().unwrap()
    }

    pub fn highest_cost(&self) -> C {
        *self.costs.iter().max().unwrap()
    }
}

pub trait Orderable = Clone + Debug + Ord + Eq;

pub trait Atom = Copy + Clone + Debug + Ord + Eq;

pub trait Cost = Num + Ord + PartialOrd + Copy + Debug;

pub trait Operator : Atom {
    type S:Clone;

    fn apply(&self, state: &Self::S) -> Option<Self::S> {
        let mut updated = state.clone();
        let success = self.attempt_update(&mut updated);
        if success {Some(updated)} else {None}
    }

    fn attempt_update(&self, state: &mut Self::S) -> bool;
}

pub enum MethodResult<O:Atom,T:Atom> {
    PlanFound,
    TaskLists(Vec<Vec<Task<O,T>>>),
    Failure
}

pub trait Method : Atom {
    type S;
    type G;
    type O: Atom;
    fn apply(&self, state: &Self::S, goal: &Self::G) -> MethodResult<Self::O, Self>;
}

pub trait Goal : Clone {
    type O: Operator;
    type M: Atom;
    fn starting_tasks(&self) -> Vec<Task<Self::O,Self::M>>;
}

#[derive(Copy,Clone,Debug)]
pub enum Task<O:Atom, T:Atom> {
    Operator(O),
    Method(T)
}

#[derive(Clone)]
struct PlannerStep<S,O,M>
where S:Orderable, O:Operator, M:Method {
    verbose: usize,
    state: S,
    prev_states: TreeSet<S>,
    tasks: Vec<Task<O,M>>,
    plan: Vec<O>,
    depth: usize
}

impl <S,O,G,M> PlannerStep<S,O,M>
    where S:Orderable, O:Operator<S=S>, G:Goal<M=M,O=O>, M:Method<S=S,G=G,O=O> {

    pub fn new(state: &S, tasks: &Vec<Task<O,M>>, verbose: usize) -> Self {
        PlannerStep {verbose, state: state.clone(), prev_states: TreeSet::new().insert(state.clone()), tasks: tasks.clone(), plan: vec![], depth: 0}
    }

    pub fn is_complete(&self) -> bool {
        self.tasks.len() == 0
    }

    pub fn get_next_step(&self, goal: &G) -> Vec<Self> {
        self.verb(format!("depth {} tasks {:?}", self.depth, self.tasks), 2);
        if self.is_complete() {
            self.verb(format!("depth {} returns plan {:?}", self.depth, self.plan), 3);
            vec![self.clone()]
        } else {
            if let Some(task1) = self.tasks.get(0) {
                match task1 {
                    Task::Operator(op) => self.apply_operator(*op),
                    Task::Method(tag) => self.apply_method(*tag, goal)
                }
            } else {
                self.verb(format!("Depth {} returns failure", self.depth), 3);
                vec![]
            }
        }
    }

    fn apply_operator(&self, operator: O) -> Vec<Self> {
        if let Some(new_state) = operator.apply(&self.state) {
            if self.prev_states.contains(&new_state) {
                self.verb(format!("Cycle; pruning..."), 3);
            } else {
                self.verb(format!("Depth {}; new_state: {:?}", self.depth, new_state), 3);
                return vec![self.operator_planner_step(new_state, operator)];
            }
        }
        vec![]
    }

    fn apply_method(&self, candidate: M, goal: &G) -> Vec<Self> {
        let mut planner_steps = Vec::new();
        match candidate.apply(&self.state, goal) {
            MethodResult::PlanFound => planner_steps.push(self.method_planner_step(&vec![])),
            MethodResult::Failure => { self.verb(format!("No plan found by method {:?}", candidate), 3); },
            MethodResult::TaskLists(subtask_alternatives) => {
                self.verb(format!("{} alternative subtask lists", subtask_alternatives.len()), 3);
                for subtasks in subtask_alternatives.iter() {
                    self.verb(format!("depth {} new tasks: {:?}", self.depth, subtasks), 3);
                    planner_steps.push(self.method_planner_step(subtasks));
                }
            }
        }
        planner_steps
    }

    fn operator_planner_step(&self, state: S, operator: O) -> Self {
        let mut updated_plan = self.plan.clone();
        updated_plan.push(operator);
        PlannerStep { verbose: self.verbose, prev_states: self.prev_states.insert(state.clone()), state: state, tasks: self.tasks[1..].to_vec(), plan: updated_plan, depth: self.depth + 1 }
    }

    fn method_planner_step(&self, subtasks: &Vec<Task<O,M>>) -> Self {
        let mut updated_tasks = Vec::new();
        subtasks.iter().for_each(|t| updated_tasks.push(*t));
        self.tasks.iter().skip(1).for_each(|t| updated_tasks.push(*t));
        PlannerStep {verbose: self.verbose, prev_states: self.prev_states.clone(), state: self.state.clone(), tasks: updated_tasks, plan: self.plan.clone(), depth: self.depth + 1}
    }

    fn verb(&self, text: String, level: usize) {
        if self.verbose > level {
            println!("{}", text);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{find_first_plan, Atom, Goal, AnytimePlannerBuilder};
    use rust_decimal_macros::*;
    use locations::LocationGraph;

    mod simple_travel;

    #[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
    enum Location {
        Home, Park, TaxiStand
    }

    #[test]
    fn simple_travel_1() {
        use crate::tests::simple_travel::{TravelState, TravelGoal, CityOperator};
        use Location::*; use CityOperator::*;
        let locations = LocationGraph::new(vec![(Home, Park, 8)]);
        let mut state = TravelState::new(locations, TaxiStand);
        state.add_traveler('M', dec!(20), Home);
        let goal = TravelGoal::new(vec![('M', Park)]);
        let tasks = goal.starting_tasks();
        let plan = find_first_plan(&state, &goal, &tasks, 3).unwrap();
        println!("the plan: {:?}", &plan);
        assert_eq!(plan, vec![(CallTaxi('M')), (RideTaxi('M', Home, Park)), (Pay('M'))]);
    }

    #[test]
    fn test_anytime() {
        use crate::tests::simple_travel::{TravelState, TravelGoal, CityOperator};
        use Location::*; use CityOperator::*;
        let locations = LocationGraph::new(vec![(Home, Park, 8)]);
        let mut state = TravelState::new(locations, TaxiStand);
        state.add_traveler('M', dec!(20), Home);
        let goal = TravelGoal::new(vec![('M', Park)]);
        let outcome = AnytimePlannerBuilder::state_goal(&state, &goal)
            .construct();
        let plan = outcome.get_best_plan();
        println!("all plans: {:?}", outcome.get_all_plans());
        println!("the plan: {:?}", &plan);
        // TODO: With the domain as stated, this is the correct low-cost plan.
        //  To make the test more interesting, we should have a cost function
        //  that uses a time metric. That cost function would no doubt prefer
        //  to drive.
        //  We could even have a third cost function regarding money, and spending
        //  very little of it.
        //  But all this is not worth it right now.
        assert_eq!(plan, vec![Walk('M', Home, Park)]);
    }
}
