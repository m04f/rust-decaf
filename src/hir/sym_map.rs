use crate::{
    hir::{
        ast::{FunctionSig, HIRFunction, HIRVar},
        error::Error::{self, *},
    },
    parser::ast::{PImport, PVar},
    span::*,
};

use std::collections::{HashMap, HashSet};

pub type SymMap<'a, T> = HashMap<Span<'a>, T>;
pub type VarSymMap<'a> = SymMap<'a, HIRVar<'a>>;
pub type FuncSymMap<'a> = SymMap<'a, HIRFunction<'a>>;
pub type ImportSymMap<'a> = HashSet<Span<'a>>;
pub type SigSymMap<'a> = SymMap<'a, FunctionSig<'a>>;

#[derive(Debug)]
pub struct SymTable<'a, 'b, O> {
    map: &'b HashMap<Span<'a>, O>,
    parent: Option<&'b Self>,
}

impl<'a, 'b, O> Clone for SymTable<'a, 'b, O> {
    fn clone(&self) -> Self {
        Self {
            map: self.map,
            parent: self.parent,
        }
    }
}

impl<'a, 'b, O> Copy for SymTable<'a, 'b, O> {}

pub type FSymMap<'a, 'b> = SymTable<'a, 'b, FunctionSig<'a>>;
pub type VSymMap<'a, 'b> = SymTable<'a, 'b, HIRVar<'a>>;

impl<'a, 'b, O> SymTable<'a, 'b, O> {
    pub fn new(map: &'b HashMap<Span<'a>, O>) -> Self {
        Self { map, parent: None }
    }

    pub fn parent(self, parent: &'b Self) -> Self {
        Self {
            map: self.map,
            parent: Some(parent),
        }
    }

    pub fn get_sym(&self, sym: Span<'a>) -> Option<&O> {
        if let Some(parent) = self.parent {
            self.map.get(&sym).or_else(|| parent.get_sym(sym))
        } else {
            self.map.get(&sym)
        }
    }
}

pub(super) fn construct_sig_hashmap<'a>(
    externs: &[PImport<'a>],
) -> Result<SigSymMap<'a>, Vec<Error<'a>>> {
    let mut errors = vec![];
    for i in 0..externs.len() {
        for j in 0..i {
            if externs[i].name() == externs[j].name() {
                errors.push(Redifinition(
                    externs[i].name().span(),
                    externs[j].name().span(),
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(externs
            .iter()
            .map(|f| (f.name().span(), FunctionSig::from_pimport(f)))
            .collect())
    } else {
        Err(errors)
    }
}

pub(super) fn construct_var_hashmap<'a, T: AsRef<[PVar<'a>]>>(
    vars: T,
) -> Result<VarSymMap<'a>, Vec<Error<'a>>> {
    let mut errors = vec![];
    let vars = vars.as_ref();
    for i in 0..vars.len() {
        for j in 0..i {
            if vars[i].name() == vars[j].name() {
                errors.push(Redifinition(vars[i].name(), vars[j].name()));
            }
        }
    }
    let vars = vars
        .iter()
        .filter_map(|var| {
            HIRVar::from_pvar(*var)
                .map(|var| (var.name(), var))
                .map_err(|e| errors.push(e))
                .ok()
        })
        .collect();
    if !errors.is_empty() {
        Err(errors)
    } else {
        Ok(vars)
    }
}

pub fn intersect<'a, O2>(
    map1: impl Iterator<Item = Span<'a>>,
    map2: &SymMap<'a, O2>,
) -> Vec<Error<'a>> {
    map1.filter_map(|span| map2.get(&span).map(|_| Redifinition(span, span)))
        .collect()
}
