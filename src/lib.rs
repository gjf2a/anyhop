use immutable_map::TreeSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::collections::VecDeque;
use std::time::Instant;
use num_traits::Num;

mod locations;

// Did not work. Was worth a try. May try again.
// Interesting discussion: https://users.rust-lang.org/t/how-to-create-a-macro-to-impl-a-provided-type-parametrized-trait/5289
// Also read this: https://danielkeep.github.io/tlborm/book/README.html
// macro_rules! num {() => {Num+Ord+PartialOrd+Copy+Debug}}
// macro_rules! num {($t:ident) => {$t:Num+Ord+PartialOrd+Copy+Debug}}

pub fn find_first_plan<S,G,O,M,T>(state: &S, goal: &G, tasks: &Vec<Task<O,T>>, verbose: usize) -> Option<Vec<O>>
    where S:Orderable, G:Goal<S,G,O,M,T>, O:Operator<S>,
          M:Method<S,G,O,M,T>,
          T:MethodTag<S,G,O,M,T> {
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

/*
// See https://doc.rust-lang.org/1.0.0/style/ownership/builders.html
// Do this later; I can always make AnytimePlanner::new() private
// to mandate the Builder.
//
// I started work on this, and I got frustrated that any default for the cost
// function overly constrains the parameterized type C.
//
// See https://stackoverflow.com/questions/32053402/why-am-i-getting-parameter-is-never-used-e0392
// to deal better with the generic types.
pub struct AnytimePlannerBuilder<'a,S,G,O,M,T,C,F>
    where S:Orderable, G:Goal<S,G,O,M,T>, O:Operator<S>,
          M:Method<S,G,O,M,T>,
          T:MethodTag<S,G,O,M,T>,
          C:Num+Ord+PartialOrd+Copy+Debug,
          F:Fn(&Vec<O>) -> C {
    state: S, goal: G, time_limit_ms: Option<u128>, strategy: BacktrackStrategy, cost_func: &'a F,
    verbose: usize, apply_cutoff: bool
}

impl <'a,S,G,O,M,T,C,F> AnytimePlannerBuilder<'a,S,G,O,M,T,C,F>
    where S:Orderable, G:Goal<S,G,O,M,T>, O:Operator<S>,
          M:Method<S,G,O,M,T>,
          T:MethodTag<S,G,O,M,T>,
          C:Num+Ord+PartialOrd+Copy+Debug,
          F:Fn(&Vec<O>) -> C {
    pub fn new(state: &S, goal: &G, cost_func: &F) -> Self {
        AnytimePlannerBuilder { state, goal, time_limit_ms: None,
            strategy: BacktrackStrategy::Steady(BacktrackPreference::MostRecent),
            cost_func, verbose: 0, apply_cutoff: true
        }
    }
}
*/

pub struct AnytimePlanner<S,G,O,M,T,C>
    where S:Orderable, G:Goal<S,G,O,M,T>, O:Operator<S>,
          M:Method<S,G,O,M,T>,
          T:MethodTag<S,G,O,M,T>,
          C:Num+Ord+PartialOrd+Copy+Debug {
    plans: Vec<Vec<O>>,
    discovery_times: Vec<u128>,
    discovery_pushes: Vec<usize>,
    discovery_pops: Vec<usize>,
    discovery_iterations: Vec<usize>,
    discovery_prunes: Vec<usize>,
    costs: Vec<C>,
    cheapest: Option<C>,
    total_iterations: usize,
    total_pops: usize,
    total_pushes: usize,
    total_pruned: usize,
    start_time: Instant,
    total_time: Option<u128>,
    current_step: PlannerStep<S,G,O,M,T>
}


pub fn summary_csv_header() -> String {
    format!("label,cheapest,discovery_time,discovery_iteration,discovery_push,discovery_pop,pruned_prior,total_time,total_iterations,total_attempts\n")
}

impl <S,G,O,M,T,C> AnytimePlanner<S,G,O,M,T,C>
    where S:Orderable, G:Goal<S,G,O,M,T>, O:Operator<S>,
          M:Method<S,G,O,M,T>,
          T:MethodTag<S,G,O,M,T>,
          C:Num+Ord+PartialOrd+Copy+Debug {
    pub fn plan<F:Fn(&Vec<O>) -> C>(state: &S, goal: &G, time_limit_ms: Option<u128>, strategy: BacktrackStrategy, cost_func: &F, verbose: usize, apply_cutoff: bool) -> Self {
        let mut outcome = AnytimePlanner {
            plans: Vec::new(), discovery_times: Vec::new(), cheapest: None, costs: Vec::new(),
            discovery_iterations: Vec::new(), discovery_pushes: Vec::new(),
            discovery_pops: Vec::new(), discovery_prunes: Vec::new(), total_iterations: 0,
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
                self.total_pruned += 1;
                (true, backtrack.1.next())
            } else {
                self.add_choices(goal, backtrack.1, &mut choices, cost_func)
            };
            if choices.is_empty() {
                self.total_time = Some(Instant::now().duration_since(self.start_time).as_millis());
                self.current_step.verb(format!("** No plans left to be found ({} ms elapsed) **", self.total_time.unwrap()), 0);
                self.current_step.verb(format!("{} attempts, {} found, {} pruned", self.plans.len() + self.total_pruned, self.plans.len(), self.total_pruned), 0);
                break;
            } else if self.time_up(time_limit_ms) {
                self.current_step.verb(format!("Time's up! {:?} ms elapsed", time_limit_ms), 0);
                break;
            } else {
                self.pick_choice(backtrack, &mut choices);
            }
        }
    }

    fn current_too_expensive<F:Fn(&Vec<O>) -> C>(&self, cost_func: F) -> bool {
        self.cheapest.map_or(false, |bound| cost_func(&self.current_step.plan) >= bound)
    }

    fn time_up(&self, time_limit_ms: Option<u128>) -> bool {
        match time_limit_ms {
            None => false,
            Some(limit) => {
                let elapsed = Instant::now().duration_since(self.start_time);
                elapsed.as_millis() >= limit
            }
        }
    }

    fn add_choices<F:Fn(&Vec<O>) -> C>(&mut self, goal: &G, strategy: BacktrackStrategy, choices: &mut VecDeque<PlannerStep<S,G,O,M,T>>, cost_func: &F) -> (bool,BacktrackStrategy) {
        if self.current_step.is_complete() {
            let plan = self.current_step.plan.clone();
            let cost: C = cost_func(&plan);
            self.cheapest = Some(self.cheapest.map_or(cost,|c| if cost < c {cost} else {c}));
            self.add_plan(self.current_step.plan.clone(), cost);
            (true, strategy.next())
        } else {
            for option in self.current_step.get_next_step(&goal) {
                choices.push_back(option);
                self.total_pushes += 1;
            }
            (false, strategy)
        }
    }

    fn add_plan(&mut self, plan: Vec<O>, cost: C) {
        self.costs.push(cost);
        let time = Instant::now().duration_since(self.start_time).as_millis();
        self.discovery_times.push(time);
        self.discovery_iterations.push(self.total_iterations);
        self.discovery_pushes.push(self.total_pushes);
        self.discovery_pops.push(self.total_pops);
        self.discovery_prunes.push(self.total_pruned);
        self.plans.push(plan);
        self.current_step.verb(format!("Plan found. Cost: {:?}; Time: {}", cost, time), 0);
    }

    fn pick_choice(&mut self, backtrack: (bool, BacktrackStrategy), choices: &mut VecDeque<PlannerStep<S,G,O,M,T>>) {
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

    pub fn report(&self) -> String {
        let c = self.index_of_cheapest();
        format!("{} plans\ncosts: {:?} ({:?})\ntimes: {} ({:?})\niterations: {} ({:?})\npushes: {} ({:?})\npops: {} ({:?})\npruned: {} ({:?})\n",
                self.plans.len(), self.lowest_cost(), &self.costs[0..c+1], self.discovery_times.last().unwrap(), &self.discovery_times[0..c+1], self.total_iterations, &self.discovery_iterations[0..c+1], self.total_pushes, &self.discovery_pushes[0..c+1], self.total_pops, &self.discovery_pops[0..c+1], self.total_pruned, &self.discovery_prunes[0..c+1])
    }

    pub fn summary_csv_row(&self, label: &str) -> String {
        format!("{},{:?},{},{},{},{},{},{},{},{}\n", label, self.lowest_cost(), self.discovery_times.last().unwrap(), self.discovery_iterations.last().unwrap(), self.discovery_pushes.last().unwrap(), self.discovery_pops.last().unwrap(), self.discovery_prunes.last().unwrap(), self.total_time.unwrap(), self.total_iterations, self.total_pruned + self.plans.len())
    }

    pub fn instance_csv(&self) -> String {
        let mut result = String::from("plan_cost,discovery_time,discovery_iteration,discovery_push,discovery_pop,pruned_prior\n");
        for p in 0..self.index_of_cheapest() {
            result.push_str(format!("{:?},{},{},{},{},{}\n", self.costs[p], self.discovery_times[p], self.discovery_iterations[p], self.discovery_pushes[p], self.discovery_pops[p], self.discovery_prunes[p]).as_str());
        }
        result
    }

    pub fn lowest_cost(&self) -> C {
        *self.costs.iter().min().unwrap()
    }
}

pub trait Orderable : Clone + Debug + Ord + Eq {}

pub trait Atom : Copy + Clone + Debug + Ord + Eq {}

pub trait Operator<S:Clone> : Atom {
    fn apply(&self, state: &S) -> Option<S> {
        let mut updated = state.clone();
        let success = self.attempt_update(&mut updated);
        if success {Some(updated)} else {None}
    }

    fn attempt_update(&self, state: &mut S) -> bool;
}

pub enum MethodResult<O:Atom,T:Atom> {
    PlanFound,
    TaskLists(Vec<Vec<Task<O,T>>>),
    Failure
}

pub trait Method<S:Clone,G:Goal<S,G,O,M,T>,O:Operator<S>,M:Method<S,G,O,M,T>,T:MethodTag<S,G,O,M,T>> : Atom {
    fn apply(&self, state: &S, goal: &G) -> MethodResult<O, T>;
}

pub trait MethodTag<S:Clone,G:Goal<S,G,O,M,T>,O:Operator<S>,M:Method<S,G,O,M,T>,T:MethodTag<S,G,O,M,T>> : Atom {
    fn candidates(&self, state: &S, goal: &G) -> Vec<M>;
}

pub trait Goal<S:Clone,G:Goal<S,G,O,M,T>,O:Operator<S>,M:Method<S,G,O,M,T>,T:MethodTag<S,G,O,M,T>> : Clone {
    fn starting_tasks(&self) -> Vec<Task<O,T>>;
}

#[derive(Copy,Clone,Debug)]
pub enum Task<O:Atom, T:Atom> {
    Operator(O),
    MethodTag(T)
}

#[derive(Clone)]
struct PlannerStep<S,G,O,M,T>
where S:Orderable, G:Goal<S,G,O,M,T>, O:Atom+Operator<S>,
      M:Atom+Method<S,G,O,M,T>,
      T:Atom+MethodTag<S,G,O,M,T> {
    verbose: usize,
    state: S,
    prev_states: TreeSet<S>,
    tasks: Vec<Task<O,T>>,
    plan: Vec<O>,
    depth: usize,
    _ph_m: PhantomData<M>,
    _ph_g: PhantomData<G>
}

impl <S,G,O,M,T> PlannerStep<S,G,O,M,T>
    where S:Orderable, G:Goal<S,G,O,M,T>, O:Operator<S>,
          M:Method<S,G,O,M,T>,
          T:MethodTag<S,G,O,M,T> {
    pub fn new(state: &S, tasks: &Vec<Task<O,T>>, verbose: usize) -> Self {
        PlannerStep {verbose, state: state.clone(), prev_states: TreeSet::new().insert(state.clone()), tasks: tasks.clone(), plan: vec![], depth: 0, _ph_m: PhantomData, _ph_g: PhantomData }
    }

    pub fn is_complete(&self) -> bool {
        self.tasks.len() == 0
    }

    pub fn get_next_step(&self, goal: &G) -> Vec<Self> {
        self.verb(format!("depth {} tasks {:?}", self.depth, self.tasks), 1);
        if self.is_complete() {
            self.verb(format!("depth {} returns plan {:?}", self.depth, self.plan), 2);
            vec![self.clone()]
        } else {
            if let Some(task1) = self.tasks.get(0) {
                match task1 {
                    Task::Operator(op) => self.apply_operator(*op),
                    Task::MethodTag(tag) => self.apply_method(*tag, goal)
                }
            } else {
                self.verb(format!("Depth {} returns failure", self.depth), 2);
                vec![]
            }
        }
    }

    fn apply_operator(&self, operator: O) -> Vec<Self> {
        if let Some(new_state) = operator.apply(&self.state) {
            if self.prev_states.contains(&new_state) {
                self.verb(format!("Cycle; pruning..."), 2);
            } else {
                self.verb(format!("Depth {}; new_state: {:?}", self.depth, new_state), 2);
                return vec![self.operator_planner_step(new_state, operator)];
            }
        }
        vec![]
    }

    fn apply_method(&self, tag: T, goal: &G) -> Vec<Self> {
        let mut planner_steps = Vec::new();
        for candidate in tag.candidates(&self.state, goal) {
            match candidate.apply(&self.state, goal) {
                MethodResult::PlanFound => planner_steps.push(self.method_planner_step(&vec![])),
                MethodResult::Failure => {self.verb(format!("No plan found by method {:?}", candidate), 2);},
                MethodResult::TaskLists(subtask_alternatives) => {
                    self.verb(format!("{} alternative subtask lists", subtask_alternatives.len()), 2);
                    for subtasks in subtask_alternatives.iter() {
                        self.verb(format!("depth {} new tasks: {:?}", self.depth, subtasks), 2);
                        planner_steps.push(self.method_planner_step(subtasks));
                    }
                }
            }
        }
        planner_steps
    }

    fn operator_planner_step(&self, state: S, operator: O) -> Self {
        let mut updated_plan = self.plan.clone();
        updated_plan.push(operator);
        PlannerStep { verbose: self.verbose, prev_states: self.prev_states.insert(state.clone()), state: state, tasks: self.tasks[1..].to_vec(), plan: updated_plan, depth: self.depth + 1, _ph_m: PhantomData, _ph_g: PhantomData }
    }

    fn method_planner_step(&self, subtasks: &Vec<Task<O,T>>) -> Self {
        let mut updated_tasks = Vec::new();
        subtasks.iter().for_each(|t| updated_tasks.push(*t));
        self.tasks.iter().skip(1).for_each(|t| updated_tasks.push(*t));
        PlannerStep {verbose: self.verbose, prev_states: self.prev_states.clone(), state: self.state.clone(), tasks: updated_tasks, plan: self.plan.clone(), depth: self.depth + 1, _ph_m: PhantomData, _ph_g: PhantomData}
    }

    fn verb(&self, text: String, level: usize) {
        if self.verbose > level {
            println!("{}", text);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{find_first_plan, Task, Atom};
    use rust_decimal_macros::*;
    use crate::locations::LocationGraph;

    mod simple_travel;
    mod simple_travel_2;

    #[derive(Copy, Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
    enum Location {
        Home, Park, TaxiStand
    }

    impl Atom for Location {}
    impl Atom for char {}

    #[test]
    fn simple_travel_1() {
        use crate::tests::simple_travel::{TravelState, TravelGoal, CityMethodTag, CityOperator};
        use Location::*; use CityOperator::*; use CityMethodTag::*;
        let locations = LocationGraph::new(vec![(Home, Park, 8)]);
        let mut state = TravelState::new(locations, TaxiStand);
        state.add_traveler('M', dec!(20), Home);
        let goal = TravelGoal::new(vec![('M', Park)]);
        let tasks = vec![Task::MethodTag(Travel('M'))];
        let plan = find_first_plan(&state, &goal, &tasks, 3).unwrap();
        println!("the plan: {:?}", &plan);
        assert_eq!(plan, vec![(CallTaxi('M')), (RideTaxi('M', Home, Park)), (Pay('M'))]);
    }

    #[test]
    fn simple_travel_2() {
        use crate::tests::simple_travel_2::{TravelState, TravelGoal, CityMethodTag, CityOperator};
        use Location::*; use CityOperator::*; use CityMethodTag::*;
        let locations = LocationGraph::new(vec![(Home, Park, 8)]);
        let mut state = TravelState::new(locations, TaxiStand);
        state.add_traveler('M', dec!(20), Home);
        let goal = TravelGoal::new(vec![('M', Park)]);
        let tasks = vec![Task::MethodTag(Travel('M'))];
        let plan = find_first_plan(&state, &goal, &tasks, 3).unwrap();
        println!("the plan: {:?}", &plan);
        assert_eq!(plan, vec![(CallTaxi('M')), (RideTaxi('M', Home, Park)), (Pay('M'))]);
    }
}
