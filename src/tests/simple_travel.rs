use rust_decimal_macros::*;
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use crate::{Task, Operator, Method, MethodTag, Atom, Orderable, LocationGraph};

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
pub enum CityMethod<T:Atom> {
    TravelByFoot(T),
    TravelByTaxi(T)
}

impl <T:Atom> Atom for CityMethod<T> {}

impl <T:Atom,L:Atom> Method<TravelState<T,L>,CityOperator<T,L>,CityMethod<T>,CityMethodTag<T>> for CityMethod<T> {
    fn apply(&self, state: &TravelState<T,L>, goal: &TravelState<T,L>) -> Vec<Vec<Task<CityOperator<T,L>, CityMethodTag<T>>>> {
        use CityOperator::*; use CityMethod::*; use Task::*;
        match self {
            TravelByFoot(t) => vec![vec![Operator(Walk(*t, state.get_location(*t).unwrap(), goal.get_location(*t).unwrap()))]],
            TravelByTaxi(t) => vec![vec![Operator(CallTaxi(*t)),
                                             Operator(RideTaxi(*t, state.get_location(*t).unwrap(), goal.get_location(*t).unwrap())),
                                             Operator(Pay(*t))]]
        }
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityMethodTag<T:Atom> {
    Travel(T)
}

impl <T:Atom> Atom for CityMethodTag<T> {}

impl <T:Atom,L:Atom> MethodTag<TravelState<T,L>,CityOperator<T,L>,CityMethod<T>,CityMethodTag<T>> for CityMethodTag<T> {
    fn candidates(&self) -> Vec<CityMethod<T>> {
        match self {
            CityMethodTag::Travel(t) =>
                vec![CityMethod::TravelByFoot(*t),
                     CityMethod::TravelByTaxi(*t)]
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

    pub fn get_location(&self, traveler: T) -> Option<L> {
        self.loc.get(&traveler).map(|l| *l)
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