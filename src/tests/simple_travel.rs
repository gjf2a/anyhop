use rust_decimal_macros::*;
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use crate::{Task, Operator, Method, MethodTag};

#[derive(Copy,Clone,Debug)]
enum CityOperator {
    Walk,
    CallTaxi,
    RideTaxi,
    Pay
}

impl <T:Copy+Ord+Eq,L:Copy+Ord+Eq> Operator<TravelState<T,L>,Args<T,L>> for CityOperator {
    fn apply(&self, state: &TravelState<T,L>, args: Args<T,L>) -> Option<TravelState<T,L>> {
        let mut updated = state.clone();
        let success = match self {
            CityOperator::Walk => updated.walk(args),
            CityOperator::CallTaxi=> updated.call_taxi(args),
            CityOperator::RideTaxi => updated.ride_taxi(args),
            CityOperator::Pay => updated.pay_driver(args)
        };
        if success {Some(updated)} else {None}
    }
}

#[derive(Copy,Clone,Debug)]
enum CityMethod {
    TravelByFoot,
    TravelByTaxi
}

impl <T:Copy+Ord+Eq,L:Copy+Ord+Eq> Method<TravelState<T,L>,Args<T,L>,CityOperator,CityMethod,CityMethodTag> for CityMethod {
    fn apply(&self, args: Args<T,L>) -> Vec<Vec<Task<CityOperator, CityMethodTag,Args<T,L>>>> {
        if let Args::Move(t, _, _) = args {
            match self {
                CityMethod::TravelByFoot =>
                    vec![vec![Task::Operator(CityOperator::Walk, args)]],
                CityMethod::TravelByTaxi =>
                    vec![vec![Task::Operator(CityOperator::CallTaxi, Args::Traveler(t)),
                              Task::Operator(CityOperator::RideTaxi, args),
                              Task::Operator(CityOperator::Pay, Args::Traveler(t))]]
            }
        } else {vec![]}
    }
}

#[derive(Copy,Clone,Debug)]
enum CityMethodTag {
    Travel
}

impl <T:Copy+Ord+Eq,L:Copy+Ord+Eq> MethodTag<TravelState<T,L>,Args<T,L>,CityOperator,CityMethod,CityMethodTag> for CityMethodTag {
    fn candidates(&self) -> Vec<CityMethod> {
        match self {
            CityMethodTag::Travel => vec![CityMethod::TravelByFoot, CityMethod::TravelByTaxi]
        }
    }
}

#[derive(Clone,PartialOrd,Ord,PartialEq,Eq,Debug)]
pub struct TravelState<T,L> {
    loc: BTreeMap<T,L>,
    cash: BTreeMap<T,Decimal>,
    owe: BTreeMap<T,Decimal>,
    taxi: L,
    dist: BTreeMap<L,BTreeMap<L,usize>>
}

#[derive(Copy,Clone,Debug)]
pub enum Args<T:Copy+Clone,L:Copy+Clone> {
    Move(T, L, L),
    Traveler(T)
}

pub fn fare(dist: usize) -> Decimal {
    dec!(1.5) + dec!(0.5) * Decimal::from(dist)
}

impl<T: Copy+Ord+Eq, L: Copy+Ord+Eq> TravelState<T,L> {
    pub fn new(dist: BTreeMap<L,BTreeMap<L,usize>>, taxi_stand: L) -> Self {
        TravelState {
            loc: BTreeMap::new(), cash: BTreeMap::new(), owe: BTreeMap::new(),
            dist: dist, taxi: taxi_stand}
    }

    pub fn get_dist(&self, start: L, end: L) -> Option<usize> {
        self.dist.get(&start)
            .and_then(|m| m.get(&end).map(|n| *n))
    }

    pub fn add_traveler(&mut self, traveler: T, cash: Decimal, start: L) {
        self.loc.insert(traveler, start);
        self.cash.insert(traveler, cash);
        self.owe.insert(traveler, dec!(0));
    }

    pub fn walk(&mut self, args: Args<T,L>) -> bool {
        if let Args::Move(t, start, end) = args {
            if let Some(at) = self.loc.get(&t) {
                if *at == start {
                    self.loc.insert(t, end);
                    return true;
                }
            }
        }
        false
    }

    pub fn call_taxi(&mut self, args: Args<T,L>) -> bool {
        if let Args::Traveler(t) = args {
            if let Some(at) = self.loc.get(&t) {
                self.taxi = *at;
                return true;
            }
        }
        false
    }

    pub fn ride_taxi(&mut self, args: Args<T,L>) -> bool {
        if let Args::Move(t, start, end) = args {
            if let Some(at) = self.loc.get(&t) {
                if *at == start && self.taxi == start {
                    if let Some(dist) = self.get_dist(start, end) {
                        self.loc.insert(t, end);
                        self.owe.insert(t, fare(dist));
                        self.taxi = end;
                        return true;
                    }
                }
            }
        }
        false
    }

    pub fn pay_driver(&mut self, args: Args<T,L>) -> bool {
        if let Args::Traveler(t) = args {
            if let Some(balance) = self.get_remaining_balance(t) {
                self.cash.insert(t, balance);
                self.owe.insert(t, dec!(0));
                return true
            }
        }
        false
    }

    pub fn get_remaining_balance(&self, traveler: T) -> Option<Decimal> {
        if let Some(cash) = self.cash.get(&traveler) {
            if let Some(owe) = self.owe.get(&traveler) {
                let diff = cash - owe;
                if diff >= dec!(0) {
                    return Some(diff);
                }
            }
        }
        None
    }
}