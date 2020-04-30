// This version of simple_travel moves away from
// using assertional states to model the world.
//
// Instead, it illustrates world modeling using more traditional data structures.
// It is a bit longer but perhaps more clear.

use rust_decimal_macros::*;
use crate::{Task, Operator, Method, MethodTag, Atom, Orderable, MethodResult};
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use crate::locations::LocationGraph;

// It is this struct that diverges from the original simple_travel.
// Specifically, instead of separate dictionaries for each aspect
// of a Traveler, all of those aspects have been consolidated into
// a Traveler struct.
//
#[derive(Clone,PartialOrd,Ord,PartialEq,Eq,Debug)]
pub struct TravelState<T:Atom,L:Atom> {
    travelers: BTreeMap<T,Traveler<L>>,
    taxi: L,
    dist: LocationGraph<L>
}

impl<T:Atom,L:Atom> Orderable for TravelState<T,L> {}

pub fn fare(dist: usize) -> Decimal {
    dec!(1.5) + dec!(0.5) * Decimal::from(dist)
}

impl<T:Atom, L:Atom> TravelState<T,L> {
    pub fn new(dist: LocationGraph<L>, taxi_stand: L) -> Self {
        TravelState {
            travelers: BTreeMap::new(),
            dist,
            taxi: taxi_stand
        }
    }

    pub fn get_dist(&self, start: L, end: L) -> Option<usize> {
        self.dist.get(start, end)
    }

    pub fn get_location(&self, traveler: T) -> Option<L> {
        self.travelers.get(&traveler).map(|t| t.location)
    }

    pub fn add_traveler(&mut self, traveler: T, cash: Decimal, start: L) {
        self.travelers.insert(traveler, Traveler {location: start, cash, owe: dec!(0)});
    }

    pub fn walk(&mut self, t: T, start: L, end: L) -> bool {
        if let Some(traveler) = self.travelers.get_mut(&t) {
            if traveler.location == start {
                traveler.location = end;
                return true;
            }
        }
        false
    }

    pub fn call_taxi(&mut self, t: T) -> bool {
        if let Some(traveler) = self.travelers.get(&t) {
            self.taxi = traveler.location;
            true
        } else { false }
    }

    pub fn ride_taxi(&mut self, t: T, start: L, end: L) -> bool {
        if let Some(dist) = self.get_dist(start, end) {
            if let Some(traveler) = self.travelers.get_mut(&t) {
                if traveler.location == start && self.taxi == start {
                    traveler.location = end;
                    self.taxi = end;
                    traveler.owe = fare(dist);
                    return true;
                }
            }
        }
        false
    }

    pub fn pay_driver(&mut self, t: T) -> bool {
        if let Some(traveler) = self.travelers.get_mut(&t) {
            if let Some(balance) = traveler.get_remaining_balance() {
                traveler.pay(balance);
                return true;
            }
        }
        false
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct Traveler<L:Atom> {
    location: L,
    cash: Decimal,
    owe: Decimal
}

impl <L:Atom> Traveler<L> {
    pub fn get_remaining_balance(&self) -> Option<Decimal> {
        let diff = self.cash - self.owe;
        if diff >= dec!(0) {Some(diff)} else {None}
    }

    pub fn pay(&mut self, amount: Decimal) {
        self.cash -= amount;
        self.owe -= amount;
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityOperator<T:Atom,L:Atom> {
    Walk(T,L,L),
    CallTaxi(T),
    RideTaxi(T,L,L),
    Pay(T)
}

impl <T:Atom,L:Atom> Atom for CityOperator<T,L> {}

impl <T:Atom,L:Atom> Operator<TravelState<T,L>> for CityOperator<T,L> {
    fn attempt_update(&self, updated: &mut TravelState<T, L>) -> bool {
        use CityOperator::*;
        match self {
            Walk(t, start, end) => updated.walk(*t, *start, *end),
            RideTaxi(t, start, end) => updated.ride_taxi(*t, *start, *end),
            CallTaxi(t) => updated.call_taxi(*t),
            Pay(t) => updated.pay_driver(*t)
        }
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityMethod<T:Atom,L:Atom> {
    TravelByFoot(T,L,L),
    TravelByTaxi(T,L,L)
}

impl <T:Atom,L:Atom> Atom for CityMethod<T,L> {}

impl <T:Atom,L:Atom> Method<TravelState<T,L>,TravelGoal<T,L>,CityOperator<T,L>,CityMethod<T,L>,CityMethodTag<T>> for CityMethod<T,L> {
    fn apply(&self, _state: &TravelState<T,L>, _goal: &TravelGoal<T,L>)
             -> MethodResult<CityOperator<T,L>, CityMethodTag<T>> {
        use CityOperator::*; use CityMethod::*; use Task::*;
        MethodResult::TaskLists(match self {
            TravelByFoot(t, start, end) => vec![vec![Operator(Walk(*t, *start, *end))]],
            TravelByTaxi(t, start, end) => vec![vec![Operator(CallTaxi(*t)),
                                                     Operator(RideTaxi(*t, *start, *end)),
                                                     Operator(Pay(*t))]]
        })
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityMethodTag<T:Atom> {
    Travel(T)
}

impl <T:Atom> Atom for CityMethodTag<T> {}

impl <T:Atom,L:Atom> MethodTag<TravelState<T,L>,TravelGoal<T,L>,CityOperator<T,L>,CityMethod<T,L>,CityMethodTag<T>> for CityMethodTag<T> {
    fn candidates(&self, state: &TravelState<T,L>, goal: &TravelGoal<T,L>) -> Vec<CityMethod<T,L>> {
        match self {
            CityMethodTag::Travel(t) =>
                if let (Some(start), Some(end)) = (state.get_location(*t), goal.goal_for(*t)) {
                    vec![CityMethod::TravelByFoot(*t, start, end),
                         CityMethod::TravelByTaxi(*t, start, end)]
                } else {
                    vec![]
                }
        }
    }
}

#[derive(Clone,Debug)]
pub struct TravelGoal<T:Atom,L:Atom> {
    goals: BTreeMap<T,L>
}

impl <T:Atom,L:Atom> TravelGoal<T,L> {
    pub fn new(goals: Vec<(T,L)>) -> Self {
        let mut result = TravelGoal {goals: BTreeMap::new()};
        for (traveler, goal) in goals {
            result.goals.insert(traveler, goal);
        }
        result
    }

    pub fn goal_for(&self, traveler: T) -> Option<L> {
        self.goals.get(&traveler).map(|g| *g)
    }
}