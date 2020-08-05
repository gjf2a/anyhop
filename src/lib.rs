#![feature(trait_alias)]
#![feature(map_first_last)]

#[macro_use]
extern crate log;

mod reflective_searcher;

use immutable_map::TreeSet;
use std::fmt::Debug;
use std::collections::{VecDeque, BTreeMap};
use std::time::Instant;
use std::{io, fs, env};
use std::fs::File;
use std::io::Write;
use reflective_searcher::TwoStageQueue;
use std::ops::Add;

pub fn find_first_plan<S,G,O,M,C>(state: &S, goal: &G, tasks: &Vec<Task<O,M>>, verbose: usize) -> Option<Vec<O>>
    where S:Orderable, G:Goal<S=S,M=M,O=O,C=C>, O:Operator<S=S,C=C,G=G>, M:Method<S=S,G=G,O=O>, C:Cost {
    let mut p = PlannerStep::new(state, tasks, verbose);
    p.verb(0,format!("** anyhop, verbose={}: **\n   state = {:?}\n   tasks = {:?}", verbose, state, tasks));
    let mut choices = VecDeque::new();
    while !p.is_complete() {
        for option in p.get_next_step(goal).0 {
            choices.push_back(option);
        }
        match choices.pop_back() {
            Some(choice) => {p = choice;},
            None => {
                p.verb(0,format!("** No plan found **"));
                return None;
            }
        }
    }
    Some(p.plan)
}

#[derive(Debug,Copy,Clone,Eq,PartialEq)]
pub enum BacktrackPreference {
    MostRecent, LeastRecent, Heuristic
}

pub struct AnytimePlannerBuilder<S,G> where S:Orderable, G:Goal {
    state: S, goal: G, time_limit_ms: Option<u128>, strategy: BacktrackPreference,
    verbose: usize, apply_cutoff: bool
}

impl <S,G,O,M,C> AnytimePlannerBuilder<S,G>
    where S:Orderable, O:Operator<S=S,C=C,G=G>, G:Goal<S=S,M=M,O=O,C=C>,
          M:Method<S=S,G=G,O=O>, C:Cost {

    pub fn state_goal(state: &S, goal: &G) -> Self {
        AnytimePlannerBuilder { state: state.clone(), goal: goal.clone(), time_limit_ms: None,
            strategy: BacktrackPreference::MostRecent, verbose: 0, apply_cutoff: true
        }
    }

    pub fn verbose(&mut self, verbose: usize) -> &mut Self {
        self.verbose = verbose;
        self
    }

    pub fn time_limit_ms(&mut self, time_limit_ms: u128) -> &mut Self {
        self.time_limit_ms = Some(time_limit_ms);
        self
    }

    pub fn possible_time_limit_ms(&mut self, time_limit_ms: Option<u128>) -> &mut Self {
        self.time_limit_ms = time_limit_ms;
        self
    }

    pub fn apply_cutoff(&mut self, apply_cutoff: bool) -> &mut Self {
        self.apply_cutoff = apply_cutoff;
        self
    }

    pub fn strategy(&mut self, strategy: BacktrackPreference) -> &mut Self {
        self.strategy = strategy;
        self
    }

    pub fn construct(&self) -> AnytimePlanner<S,O,M,C,G> {
        AnytimePlanner::plan(&self.state, &self.goal, self.time_limit_ms, self.strategy, self.verbose, self.apply_cutoff)
    }
}

pub struct AnytimePlanner<S,O,M,C,G>
    where S:Orderable, O:Operator<S=S,C=C,G=G>, M:Method, C:Cost, G:Goal<S=S,M=M,O=O,C=C> {
    plans: Vec<Vec<O>>,
    flawed_plans: Vec<Vec<O>>,
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
    current_step: PlannerStep<S,O,M,C,G>,
    strategy: BacktrackPreference,
    apply_cutoff: bool
}

pub fn summary_csv_header() -> String {
    format!("label,first,most_expensive,cheapest,discovery_time,total_time,pruned_prior,num_prior_plans,total_prior_attempts,total_attempts\n")
}

#[derive(Copy,Clone,Eq,PartialEq,Debug)]
enum Backtrack {
    Yes, No
}

#[derive(Copy,Clone,Eq,PartialEq,Debug)]
enum SearchStatus {
    Failure, Cycle, Ongoing, Completed
}

impl <S,O,C,G,M> AnytimePlanner<S,O,M,C,G>
    where S:Orderable, O:Operator<S=S,C=C,G=G>, C:Cost,
          G:Goal<S=S,M=M,O=O,C=C>,
          M:Method<S=S,G=G,O=O> {
    fn plan(state: &S, goal: &G, time_limit_ms: Option<u128>, strategy: BacktrackPreference, verbose: usize, apply_cutoff: bool) -> Self {
        let mut outcome = AnytimePlanner {
            plans: Vec::new(), discovery_times: Vec::new(), cheapest: None, costs: Vec::new(),
            discovery_prior_plans: Vec::new(), discovery_prunes: Vec::new(), total_iterations: 0,
            total_pops: 0, total_pushes: 0, total_pruned: 0, start_time: Instant::now(),
            total_time: None, strategy, apply_cutoff, flawed_plans: Vec::new(),
            current_step: PlannerStep::new(state, &goal.starting_tasks(), verbose)
        };
        outcome.make_plan(goal, time_limit_ms, strategy, apply_cutoff);
        outcome
    }

    fn make_plan(&mut self, goal: &G, time_limit_ms: Option<u128>, strategy: BacktrackPreference, apply_cutoff: bool) {
        let mut choices = TwoStageQueue::new();
        self.current_step.verb(0, format!("Verbosity level: {}", self.current_step.verbose));
        self.current_step.verb(0, format!("Backtrack strategy: {:?}", strategy));
        self.current_step.verb(1, format!("Branch and bound pruning? {}", apply_cutoff));
        self.current_step.verb(3, format!("Initial state: {:?}", self.current_step.state));
        loop {
            self.total_iterations += 1;
            let backtrack = self.determine_choices(goal, apply_cutoff, &mut choices);
            if choices.is_empty() {
                self.set_total_time();
                self.current_step.verb(0,format!("** No plans left to be found ({} ms elapsed) **", self.total_time.unwrap()));
                self.current_step.verb(0,format!("{} attempts, {} found, {} pruned", self.plans.len() + self.total_pruned, self.plans.len(), self.total_pruned));
                break;
            } else if self.time_up(time_limit_ms) {
                self.set_total_time();
                self.current_step.verb(0,format!("Time's up! {:?} ms elapsed", time_limit_ms));
                break;
            } else {
                self.pick_choice(backtrack, strategy, goal, &mut choices);
            }
        }
    }

    fn determine_choices(&mut self, goal: &G, apply_cutoff: bool, choices: &mut TwoStageQueue<C,PlannerStep<S,O,M,C,G>>) -> Backtrack {
        if apply_cutoff && self.current_too_expensive() {
            let time = self.time_since_start();
            self.total_pruned += 1;
            self.current_step.verb(1,format!("Plan pruned. Time: {}", time));
            Backtrack::Yes
        } else {
            self.add_choices(goal, choices)
        }
    }

    fn time_since_start(&self) -> u128 {
        Instant::now().duration_since(self.start_time).as_millis()
    }

    fn set_total_time(&mut self) {
        self.total_time = Some(self.time_since_start());
    }

    fn current_too_expensive(&self) -> bool {
        self.cheapest.map_or(false, |bound| self.current_step.cost >= bound)
    }

    fn time_up(&self, time_limit_ms: Option<u128>) -> bool {
        match time_limit_ms {
            None => false,
            Some(limit) => self.time_since_start() >= limit
        }
    }

    fn add_choices(&mut self, goal: &G, choices: &mut TwoStageQueue<C,PlannerStep<S,O,M,C,G>>) -> Backtrack {
        use SearchStatus::*;
        let (options, status) = self.current_step.get_next_step(goal);
        match status {
            Completed => {
                self.add_plan(goal);
                Backtrack::Yes
            },
            Cycle => Backtrack::No,
            Failure => Backtrack::Yes,
            Ongoing => {
                for option in options {
                    choices.insert(option);
                    self.total_pushes += 1;
                }
                Backtrack::No
            }
        }
    }

    fn add_plan(&mut self, goal: &G) {
        let cost = self.current_step.cost;
        self.costs.push(cost);
        let time = self.time_since_start();
        self.discovery_times.push(time);
        self.discovery_prunes.push(self.total_pruned);
        self.discovery_prior_plans.push(self.plans.len());
        let plan = self.current_step.plan.clone();
        if goal.accepts(&self.current_step.state) {
            self.cheapest = Some(self.cheapest.map_or(cost,|c| if cost < c {cost} else {c}));
            self.plans.push(plan);
            self.current_step.verb(0, format!("Plan found. Cost: {:?}; Time: {}", cost, time));
        } else {
            self.flawed_plans.push(plan);
            self.current_step.verb(0, format!("Plan found, but goals are not met. Cost: {:?}; Time: {}", cost, time));
        }
    }

    fn pick_choice(&mut self, backtrack: Backtrack, strategy: BacktrackPreference, goal: &G, choices: &mut TwoStageQueue<C,PlannerStep<S,O,M,C,G>>) {
        if backtrack == Backtrack::Yes {
            match strategy {
                BacktrackPreference::MostRecent => {},
                BacktrackPreference::LeastRecent => choices.to_bfs(),
                BacktrackPreference::Heuristic => choices.to_heap(|step| step.cost + goal.distance_from(&step.state))
            }
        }
        self.current_step = choices.remove().unwrap();
        self.total_pops += 1;
    }

    pub fn index_of_cheapest(&self) -> usize {
        (0..self.costs.len())
            .fold(0, |best, i| if self.costs[i] < self.costs[best] {i} else {best})
    }

    pub fn get_plan(&self, i: usize) -> Option<Vec<O>> {
        self.plans.get(i).map(|p| p.clone())
    }

    pub fn get_best_plan(&self) -> Option<Vec<O>> {
        self.get_plan(self.index_of_cheapest())
    }

    pub fn get_all_plans(&self) -> Vec<Vec<O>> {
        self.plans.clone()
    }

    pub fn get_flawed_plans(&self) -> Vec<Vec<O>> {
        self.flawed_plans.clone()
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

pub trait Cost = Ord + PartialOrd + Copy + Debug + Add<Output=Self>;

pub trait Operator : Atom {
    type S:Clone;
    type C:Cost;
    type G:Goal<O=Self,S=Self::S,C=Self::C>;

    fn apply(&self, state: &Self::S) -> Option<Self::S> {
        let mut updated = state.clone();
        let success = self.attempt_update(&mut updated);
        if success {Some(updated)} else {None}
    }

    fn cost(&self, state: &Self::S, goal: &Self::G) -> Self::C;
    fn zero_cost() -> Self::C;
    fn attempt_update(&self, state: &mut Self::S) -> bool;
}

pub enum MethodResult<O:Atom,T:Atom> {
    TaskLists(Vec<Vec<Task<O,T>>>),
    Failure
}

pub trait Method : Atom {
    type S;
    type G;
    type O: Atom;
    fn apply(&self, state: &Self::S, goal: &Self::G) -> MethodResult<Self::O, Self>;
}

pub trait Goal : Clone + Debug {
    type O: Operator<S=Self::S,C=Self::C,G=Self>;
    type M: Atom;
    type S: Clone;
    type C: Cost;

    fn starting_tasks(&self) -> Vec<Task<Self::O,Self::M>>;
    fn accepts(&self, state: &Self::S) -> bool;
    fn distance_from(&self, state: &Self::S) -> Self::C;

    fn plan_valid(&self, start: &Self::S, plan: &Vec<Self::O>) -> bool {
        let mut state = start.clone();
        for op in plan.iter() {
            if !op.attempt_update(&mut state) {
                return false;
            }
        }
        self.accepts(&state)
    }
}

#[derive(Copy,Clone,Debug)]
pub enum Task<O:Atom, T:Atom> {
    Operator(O),
    Method(T)
}

#[derive(Clone)]
struct PlannerStep<S,O,M,C,G>
where S:Orderable, O:Operator<S=S,C=C,G=G>, M:Method, C:Cost, G:Goal<S=S,M=M,O=O,C=C>, {
    verbose: usize,
    state: S,
    prev_states: TreeSet<S>,
    tasks: Vec<Task<O,M>>,
    plan: Vec<O>,
    cost: C,
    depth: usize
}

impl <S,O,G,M,C> PlannerStep<S,O,M,C,G>
    where S:Orderable, O:Operator<S=S,C=C,G=G>, G:Goal<S=S,M=M,O=O,C=C>, M:Method<S=S,G=G,O=O>, C:Cost {

    pub fn new(state: &S, tasks: &Vec<Task<O,M>>, verbose: usize) -> Self {
        PlannerStep {verbose, state: state.clone(), prev_states: TreeSet::new().insert(state.clone()), tasks: tasks.clone(), plan: vec![], depth: 0, cost: O::zero_cost()}
    }

    pub fn is_complete(&self) -> bool {
        self.tasks.len() == 0
    }

    pub fn get_next_step(&self, goal: &G) -> (Vec<Self>,SearchStatus) {
        self.verb(2,format!("depth {} tasks {:?}", self.depth, self.tasks));
        if self.is_complete() {
            self.verb(3,format!("depth {} returns plan {:?}", self.depth, self.plan));
            (vec![self.clone()],SearchStatus::Completed)
        } else {
            if let Some(task1) = self.tasks.get(0) {
                match task1 {
                    Task::Operator(op) => self.apply_operator(*op, goal),
                    Task::Method(tag) => self.apply_method(*tag, goal)
                }
            } else {
                self.verb(3,format!("Depth {} returns failure", self.depth));
                (vec![],SearchStatus::Failure)
            }
        }
    }

    fn apply_operator(&self, operator: O, goal: &G) -> (Vec<Self>,SearchStatus) {
        if let Some(new_state) = operator.apply(&self.state) {
            if self.prev_states.contains(&new_state) {
                self.verb(3,format!("Cycle after applying operator {:?}; pruning...", operator));
                (vec![], SearchStatus::Cycle)
            } else {
                self.verb(3,format!("Depth {}; new_state: {:?}", self.depth, new_state));
                (vec![self.operator_planner_step(new_state, operator, goal)], SearchStatus::Ongoing)
            }
        } else {
            (vec![], SearchStatus::Failure)
        }
    }

    fn apply_method(&self, candidate: M, goal: &G) -> (Vec<Self>,SearchStatus) {
        let mut planner_steps = Vec::new();
        match candidate.apply(&self.state, goal) {
            MethodResult::Failure => {
                self.verb(3,format!("No plan found by method {:?}", candidate));
                self.verb(3, format!("Incomplete and inadequate plan (cost {:?}): {:?}", self.cost, self.plan));
                (planner_steps, SearchStatus::Failure)
            },
            MethodResult::TaskLists(subtask_alternatives) => {
                let num_alternatives = subtask_alternatives.len();
                if num_alternatives > 0 {
                    self.verb(3, format!("{} alternative subtask lists", num_alternatives));
                    for subtasks in subtask_alternatives.iter() {
                        self.verb(3, format!("depth {} new tasks: {:?}", self.depth, subtasks));
                        planner_steps.push(self.method_planner_step(subtasks));
                    }
                } else {
                    self.verb(3, format!("Plan found"));
                    planner_steps.push(self.method_planner_step(&vec![]));
                }
                (planner_steps, SearchStatus::Ongoing)
            }
        }
    }

    fn operator_planner_step(&self, state: S, operator: O, goal: &G) -> Self {
        let mut updated_plan = self.plan.clone();
        updated_plan.push(operator);
        PlannerStep { verbose: self.verbose, prev_states: self.prev_states.insert(state.clone()), state: state, tasks: self.tasks[1..].to_vec(), plan: updated_plan, depth: self.depth + 1, cost: self.cost + operator.cost(&self.state, goal) }
    }

    fn method_planner_step(&self, subtasks: &Vec<Task<O,M>>) -> Self {
        let mut updated_tasks = Vec::new();
        subtasks.iter().for_each(|t| updated_tasks.push(*t));
        self.tasks.iter().skip(1).for_each(|t| updated_tasks.push(*t));
        PlannerStep {verbose: self.verbose, prev_states: self.prev_states.clone(), state: self.state.clone(), tasks: updated_tasks, plan: self.plan.clone(), depth: self.depth + 1, cost: self.cost}
    }

    fn verb(&self, level: usize, text: String) {
        if self.verbose > level {
            debug!("{}", text); //This should be set to debug, not info, as to not spam the log with tasks on verbosity level 2.
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{find_first_plan, Atom, Goal, AnytimePlannerBuilder, BacktrackPreference};
    use rust_decimal_macros::*;
    use locations::LocationGraph;
    use crate::tests::simple_tsp::{TSPGoal, TSPState};

    mod simple_travel;
    mod simple_tsp;

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
        info!("the plan: {:?}", &plan);
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
            .verbose(4)
            .construct();
        let plan = outcome.get_best_plan().unwrap();
        info!("all plans: {:?}", outcome.get_all_plans());
        info!("the plan: {:?}", &plan);
        // TODO: With the domain as stated, this is the correct low-cost plan.
        //  To make the test more interesting, we should have a cost function
        //  that uses a time metric. That cost function would no doubt prefer
        //  to drive.
        //  We could even have a third cost function regarding money, and spending
        //  very little of it.
        //  But all this is not worth it right now.
        assert_eq!(plan, vec![Walk('M', Home, Park)]);
    }

    fn make_tsp_1() -> TSPGoal {
        TSPGoal::new(vec![(0, 1, 1), (0, 2, 2), (1, 2, 3)])
    }

    fn simple_order_test(pref: BacktrackPreference, target_result: &str) {
        start_logger();
        let goal = make_tsp_1();
        let state = TSPState::new(&goal);
        let outcome = AnytimePlannerBuilder::state_goal(&state, &goal)
            .strategy(pref)
            .verbose(4)
            .apply_cutoff(false)
            .construct();
        assert_eq!(format!("{:?}", outcome.get_all_plans()).as_str(), target_result);
    }

    #[test]
    fn test_dfs() {
        simple_order_test(BacktrackPreference::MostRecent, r#"[[TSPMove { end: 2 }, TSPMove { end: 1 }, TSPMove { end: 0 }], [TSPMove { end: 2 }, TSPMove { end: 0 }, TSPMove { end: 1 }], [TSPMove { end: 1 }, TSPMove { end: 2 }, TSPMove { end: 0 }], [TSPMove { end: 1 }, TSPMove { end: 0 }, TSPMove { end: 2 }], [TSPMove { end: 0 }, TSPMove { end: 2 }, TSPMove { end: 1 }], [TSPMove { end: 0 }, TSPMove { end: 1 }, TSPMove { end: 2 }]]"#);
    }

    #[test]
    fn test_bfs() {
        simple_order_test(BacktrackPreference::LeastRecent, r#"[[TSPMove { end: 2 }, TSPMove { end: 1 }, TSPMove { end: 0 }], [TSPMove { end: 0 }, TSPMove { end: 2 }, TSPMove { end: 1 }], [TSPMove { end: 1 }, TSPMove { end: 2 }, TSPMove { end: 0 }], [TSPMove { end: 2 }, TSPMove { end: 0 }, TSPMove { end: 1 }], [TSPMove { end: 0 }, TSPMove { end: 1 }, TSPMove { end: 2 }], [TSPMove { end: 1 }, TSPMove { end: 0 }, TSPMove { end: 2 }]]"#);
    }

    fn start_logger() {
        match simple_logger::init() {
            Err(_)=> {},
            _ => {}
        }
    }
}

// Experiment harness functions

pub fn process_expr_cmd_line<S,O,G,M,P,C>(parser: &P, args: &CmdArgs) -> io::Result<()>
    where S:Orderable, O:Operator<S=S,C=C,G=G>, G:Goal<S=S,M=M,O=O,C=C>, M:Method<S=S,G=G,O=O>,
          P: Fn(&str) -> io::Result<(S, G)>, C:Cost {

    let mut results = summary_csv_header();
    let (limit_ms, verbosity, apply_cutoff, strategies) = find_time_limit_verbosity_cutoff(args);
    for file in args.all_filenames().iter() {
        FileAssessor::assess_file(file.as_str(), &mut results, limit_ms, verbosity, apply_cutoff, &strategies, parser)?;
    }
    let mut output = File::create(make_result_filename(args))?;
    write!(output, "{}", results.as_str())?;
    Ok(())
}

fn make_result_filename(args: &CmdArgs) -> String {
    let mut result = String::from("results");
    result.push_str(args.get_with_tag("s")
        .map_or("_no_time_limit", |s| s.as_str()));
    args.all_filenames().iter().for_each(|s| {result.push('_'); result.push_str(simplify_filename(s.as_str()).as_str())});
    result = result.replace('.', "_").replace('-', "_");
    result.push_str(".csv");
    result
}

fn simplify_filename(filename: &str) -> String {
    std::path::Path::new(filename).file_name()
        .map_or(String::from("_missing_name"), |s| desuffix(s.to_str().unwrap()))
}

#[derive(Debug,Clone)]
pub struct CmdArgs {
    options: BTreeMap<String,String>,
    filenames: Vec<String>
}

impl CmdArgs {
    pub fn new() -> io::Result<Self> {
        let mut result = CmdArgs {options: BTreeMap::new(), filenames: Vec::new()};
        let mut args_iter = env::args().skip(1).peekable();
        while args_iter.peek().map_or(false, |s| s.starts_with("-")) {
            let arg = args_iter.next().unwrap();
            if let Some(tag) = CmdArgs::arg_tag(arg.as_str()) {
                result.options.insert(tag, arg);
            }
        }
        for file in args_iter {
            if file.ends_with("*") {
                let mut no_star = file.clone();
                no_star.pop();
                for entry in fs::read_dir(".")? {
                    let entry = entry?;
                    let entry = entry.file_name();
                    let entry = entry.to_str();
                    let entry_name = entry.unwrap();
                    if entry_name.starts_with(no_star.as_str()) {
                        result.filenames.push(String::from(entry_name));
                    }
                }
            } else {
                result.filenames.push(file);
            }
        }
        Ok(result)
    }

    fn arg_tag(arg: &str) -> Option<String> {
        if arg.len() >= 2 && arg.starts_with("-") {
            Some(arg.chars().skip_while(|c| !c.is_alphabetic()).collect())
        } else {
            None
        }
    }

    pub fn get_with_tag(&self, arg_tag: &str) -> Option<&String> {
        self.options.get(arg_tag)
    }

    pub fn has_tag(&self, arg_tag: &str) -> bool {
        self.options.contains_key(arg_tag)
    }

    pub fn num_from<N: std::str::FromStr>(&self, arg_tag: &str) -> Option<N> {
        match self.options.get(arg_tag) {
            Some(arg) => {
                let num = &arg[1..arg.len() - 1];
                match num.parse::<N>() {
                    Ok(num) => Some(num),
                    Err(_) => {println!("{} is not valid", num); None}
                }
            },
            None => None
        }
    }

    pub fn all_filenames(&self) -> &Vec<String> {
        &self.filenames
    }

    pub fn all_options(&self) -> Vec<String> {
        self.options.iter().map(|(_,v)| v.clone()).collect()
    }
}

fn find_time_limit_verbosity_cutoff(args: &CmdArgs) -> (Option<u128>,Option<usize>,bool,Vec<BacktrackPreference>) {
    if args.has_tag("h") || args.has_tag("help") {
        println!("Usage: planner [-h] [-c] [-(int)s] [[-(int)v] plan_files");
        println!("\t-h: This message");
        println!("\t-c: See command-line argument data structure");
        println!("\t-no_prune: No branch-and-bound cutoff");
        println!("\t-dfs: Depth-first only");
        println!("\t-bfs: Breadth-first only");
        println!("\t-heu: Heuristic search only");
        println!("\t-(int)s: Time limit in seconds (e.g. -5s => 5 seconds)");
        println!("\t-(int)v: Verbosity (0-4)");
        println!("\t\t-0v: Reports final plan only");
        println!("\t\t-1v: Reports plan found, time limit reached, no more plans to be found");
        println!("\t\t-2v: Reports branch-and-bound pruning");
        println!("\t\t-3v: Reports tasks at each depth level reached");
        println!("\t\t-4v: Reports the following at each new depth level:");
        println!("\t\t\tPlan found");
        println!("\t\t\tFailure");
        println!("\t\t\tPruning due to cycle");
        println!("\t\t\tNew state");
        println!("\t\t\tAlternative task lists");
    }
    if args.has_tag("c") {
        println!("CmdArgs: {:?}", args);
        println!("verbosity: {:?}; limit: {:?}", args.num_from::<usize>("v"), args.num_from::<usize>("s"));
    }
    (args.num_from("s").map(|s: u128| s * 1000),
     args.num_from("v"),
     !args.has_tag("no_prune"),
     backtrack_prefs(&args))
}

fn backtrack_prefs(args: &CmdArgs) -> Vec<BacktrackPreference> {
    let mut result = Vec::new();
    if args.has_tag("dfs") {
        result.push(BacktrackPreference::MostRecent);
    }
    if args.has_tag("bfs") {
        result.push(BacktrackPreference::LeastRecent);
    }
    if args.has_tag("heu") {
        result.push(BacktrackPreference::Heuristic);
    }
    if result.is_empty() {
        vec![BacktrackPreference::LeastRecent, BacktrackPreference::MostRecent, BacktrackPreference::Heuristic]
    } else {
        result
    }
}

pub struct FileAssessor<S,O,G,M,C>
    where S:Orderable, O:Operator<S=S,C=C,G=G>, G:Goal<S=S,M=M,O=O,C=C>,
          M:Method<S=S,G=G,O=O>, C:Cost{
    file: String,
    results: String,
    outcome: AnytimePlanner<S,O,M,C,G>
}

impl <S,O,G,M,C> FileAssessor<S,O,G,M,C>
    where S:Orderable, O:Operator<S=S,C=C,G=G>, G:Goal<S=S,M=M,O=O,C=C>,
          M:Method<S=S,G=G,O=O>, C:Cost {
    fn assess_file<P: Fn(&str) -> io::Result<(S,G)>>(file: &str, results: &mut String, limit_ms: Option<u128>, verbosity: Option<usize>, apply_cutoff: bool, strategies: &Vec<BacktrackPreference>, parser: &P) -> io::Result<()> {
        debug!("assess_file(\"{}\"): verbosity: {:?} ({:?})", file, verbosity, verbosity.unwrap_or(1));
        info!("Running {}", file);
        let (start, goal) = parser(file)?;
        debug!("Start state: {:?}", start);
        debug!("Goal: {:?}", goal);
        for strategy in strategies.iter() {
            let mut assessor = FileAssessor {
                file: String::from(file), results: String::new(),
                outcome: AnytimePlannerBuilder::state_goal(&start, &goal)
                    .apply_cutoff(apply_cutoff)
                    .strategy(*strategy)
                    .possible_time_limit_ms(limit_ms)
                    .verbose(verbosity.unwrap_or(1))
                    .construct()
            };
            assessor.report()?;
            results.push_str(assessor.results.as_str());
        }
        Ok(())
    }

    fn report(&mut self) -> io::Result<()> {
        match self.outcome.get_best_plan() {
            Some(plan) => {
                self.plan_report(&plan)?;
            },
            None => warn!("No plan found.")
        };
        let flawed = self.outcome.get_flawed_plans();
        if flawed.len() > 0 {
            warn!("{} flawed plans found.", flawed.len());
            for i in 0..flawed.len() {
                warn!("Flawed plan {}", i + 1);
                warn!("{:?}", flawed[i]);
            }
        }
        Ok(())
    }

    fn plan_report(&mut self, plan: &Vec<O>) -> io::Result<()> {
        info!("Plan (filename {:?}:", self.file);
        info!("{:?}", plan);
        let label = format!("o_{}_{:?}_{}", desuffix(self.file.as_str()), self.outcome.strategy, if self.outcome.apply_cutoff { "cutoff" } else { "no_cutoff" })
            .replace(")", "_")
            .replace("(", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_");
        let row = self.outcome.summary_csv_row(label.as_str());
        info!("{}", row);
        self.results.push_str(row.as_str());
        let mut output = File::create(format!("{}.csv", label))?;
        write!(output, "{}", self.outcome.instance_csv())?;
        Ok(())
    }
}

fn desuffix(filename: &str) -> String {
    if filename.contains(".") {
        filename.split(".").next().unwrap().to_string()
    } else {
        String::from(filename)
    }
}
