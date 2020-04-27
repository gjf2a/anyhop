use rust_decimal_macros::*;
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use crate::{Task, Operator, Method, MethodTag, Atom, Orderable, LocationGraph};

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityOperator {
    Walk,
    CallTaxi,
    RideTaxi,
    Pay
}

impl Atom for CityOperator {}

impl <T:Atom,L:Atom> Operator<TravelState<T,L>,Args<T,L>> for CityOperator {
    fn attempt_update(&self, updated: &mut TravelState<T, L>, args: Args<T, L>) -> bool {
        use CityOperator::*; use Args::*;
        match args {
            Move(t, start, end) => match self {
                Walk => updated.walk(t, start, end),
                RideTaxi => updated.ride_taxi(t, start, end),
                _ => false
            },
            Traveler(t) => match self {
                CallTaxi => updated.call_taxi(t),
                Pay => updated.pay_driver(t),
                _ => false
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityMethod {
    TravelByFoot,
    TravelByTaxi
}

impl Atom for CityMethod {}

impl <T:Atom,L:Atom> Method<TravelState<T,L>,Args<T,L>,CityOperator,CityMethod,CityMethodTag> for CityMethod {
    fn apply(&self, args: Args<T,L>) -> Vec<Vec<Task<CityOperator, CityMethodTag,Args<T,L>>>> {
        use CityOperator::*; use CityMethod::*; use Task::*;
        if let Args::Move(t, _, _) = args {
            match self {
                TravelByFoot => vec![vec![Operator(Walk, args)]],
                TravelByTaxi => vec![vec![Operator(CallTaxi, Args::Traveler(t)),
                                          Operator(RideTaxi, args),
                                          Operator(Pay, Args::Traveler(t))]]
            }
        } else {vec![]}
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityMethodTag {
    Travel
}

impl Atom for CityMethodTag {}

impl <T:Atom,L:Atom> MethodTag<TravelState<T,L>,Args<T,L>,CityOperator,CityMethod,CityMethodTag> for CityMethodTag {
    fn candidates(&self) -> Vec<CityMethod> {
        match self {
            CityMethodTag::Travel => vec![CityMethod::TravelByFoot, CityMethod::TravelByTaxi]
        }
    }
}

#[derive(Clone,PartialOrd,Ord,PartialEq,Eq,Debug)]
pub struct TravelState<T,L:Atom> {
    loc: BTreeMap<T,L>,
    cash: BTreeMap<T,Decimal>,
    owe: BTreeMap<T,Decimal>,
    taxi: L,
    dist: LocationGraph<L>
}

impl<T:Atom,L:Atom> Orderable for TravelState<T,L> {}

#[derive(Copy,Clone,Debug,Ord, PartialOrd, Eq, PartialEq)]
pub enum Args<T:Atom,L:Atom> {
    Move(T, L, L),
    Traveler(T)
}

impl<T:Atom,L:Atom> Atom for Args<T,L> {}

pub fn fare(dist: usize) -> Decimal {
    dec!(1.5) + dec!(0.5) * Decimal::from(dist)
}

impl<T:Atom, L:Atom> TravelState<T,L> {
    pub fn new(dist: LocationGraph<L>, taxi_stand: L) -> Self {
        TravelState {
            loc: BTreeMap::new(), cash: BTreeMap::new(), owe: BTreeMap::new(),
            dist: dist, taxi: taxi_stand}
    }

    pub fn get_dist(&self, start: L, end: L) -> Option<usize> {
        self.dist.get(start, end)
    }

    pub fn add_traveler(&mut self, traveler: T, cash: Decimal, start: L) {
        self.loc.insert(traveler, start);
        self.cash.insert(traveler, cash);
        self.owe.insert(traveler, dec!(0));
    }

    pub fn walk(&mut self, t: T, start: L, end: L) -> bool {
        if let Some(_) = self.loc.get(&t).filter(|at| **at == start) {
            self.loc.insert(t, end);
            true
        } else {false}
    }

    pub fn call_taxi(&mut self, t: T) -> bool {
        if let Some(at) = self.loc.get(&t) {
            self.taxi = *at;
            true
        } else {false}
    }

    pub fn ride_taxi(&mut self, t: T, start: L, end: L) -> bool {
        if let Some(dist) = self.loc.get(&t)
            .filter(|at| **at == start && self.taxi == start)
            .and_then(|_| self.get_dist(start, end)) {
            self.loc.insert(t, end);
            self.owe.insert(t, fare(dist));
            self.taxi = end;
            true
        } else {false}
    }

    pub fn pay_driver(&mut self, t: T) -> bool {
        if let Some(balance) = self.get_remaining_balance(t) {
            self.cash.insert(t, balance);
            self.owe.insert(t, dec!(0));
            true
        } else {false}
    }

    pub fn get_remaining_balance(&self, traveler: T) -> Option<Decimal> {
        if let (Some(cash), Some(owe)) = (self.cash.get(&traveler), self.owe.get(&traveler)) {
            let diff = cash - owe;
            if diff >= dec!(0) {return Some(diff);}
        }
        None
    }
}