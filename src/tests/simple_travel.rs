use rust_decimal_macros::*;
use rust_decimal::Decimal;
use std::collections::BTreeMap;
use crate::{Task, Operator, Method, MethodTag, Atom, Orderable};
use immutable_map::TreeMap;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct MapGraph<L: Atom> {
    distances: TreeMap<L,TreeMap<L,usize>>
}

impl <L:Atom> MapGraph<L> {
    pub fn new(distances: Vec<(L,L,usize)>) -> Self {
        let mut map_graph = MapGraph {distances: TreeMap::new()};
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

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityOperator {
    Walk,
    CallTaxi,
    RideTaxi,
    Pay
}

impl Atom for CityOperator {}

impl <T:Atom,L:Atom> Operator<TravelState<T,L>,Args<T,L>> for CityOperator {
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

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub enum CityMethod {
    TravelByFoot,
    TravelByTaxi
}

impl Atom for CityMethod {}

impl <T:Atom,L:Atom> Method<TravelState<T,L>,Args<T,L>,CityOperator,CityMethod,CityMethodTag> for CityMethod {
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
    dist: MapGraph<L>
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
    pub fn new(dist: MapGraph<L>, taxi_stand: L) -> Self {
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
            if let Some(dist) = self.loc.get(&t)
                .filter(|at| **at == start && self.taxi == start)
                .and_then(|_| self.get_dist(start, end)) {
                self.loc.insert(t, end);
                self.owe.insert(t, fare(dist));
                self.taxi = end;
                return true;
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
        self.cash.get(&traveler)
            .and_then(|cash| self.owe.get(&traveler).map(|owe| cash - owe))
            .filter(|diff| *diff >= dec!(0))
    }
}