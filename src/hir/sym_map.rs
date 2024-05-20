use crate::{
    hir::{
        ast::{FunctionSig, HIRFunction, HIRVar},
        error::Error::{self, *},
    },
    parser::ast::{PImport, PVar},
    span::*,
};

use std::collections::{HashMap, HashSet};

pub type SymMap<T> = HashMap<String, T>;
pub type VarSymMap = SymMap<HIRVar>;
pub type FuncSymMap = SymMap<HIRFunction>;
pub type ImportSymMap = HashSet<String>;
pub type SigSymMap = SymMap<FunctionSig>;

#[derive(Debug, Clone, Copy)]
pub struct SymTable<'a, O> {
    map: &'a HashMap<String, O>,
    parent: Option<&'a Self>,
}

pub type FSymMap<'a> = SymTable<'a, FunctionSig>;
pub type VSymMap<'a> = SymTable<'a, HIRVar>;

impl<'a, O> SymTable<'a, O> {
    pub fn new(map: &'a HashMap<String, O>) -> Self {
        Self { map, parent: None }
    }

    pub fn parent(self, parent: &'a Self) -> Self {
        Self {
            map: self.map,
            parent: Some(parent),
        }
    }

    pub fn get_sym(&self, sym: impl AsRef<str>) -> Option<&O> {
        if let Some(parent) = self.parent {
            self.map.get(sym.as_ref()).or_else(|| parent.get_sym(sym))
        } else {
            self.map.get(sym.as_ref())
        }
    }
}

pub(super) fn construct_sig_hashmap<'a>(
    externs: &[PImport<'a>],
) -> Result<SigSymMap, Vec<Error<'a>>> {
    let mut errors = vec![];
    for i in 0..externs.len() {
        for j in 0..i {
            if externs[i].name() == externs[j].name() {
                errors.push(Redifinition(
                    externs[i].name(),
                    externs[j].name(),
                ));
            }
        }
    }
    if errors.is_empty() {
        Ok(externs
            .iter()
            .map(|f| (f.name().to_string(), FunctionSig::from_pimport(f)))
            .collect())
    } else {
        Err(errors)
    }
}

pub(super) fn construct_var_hashmap<'a, T: AsRef<[PVar<'a>]>>(
    vars: T,
) -> Result<VarSymMap, Vec<Error<'a>>> {
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
                .map(|var| (var.name().to_string(), var))
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

pub fn intersect<'a>(
    map1: &HashSet<Span<'a>>,
    map2: impl Iterator<Item = Span<'a>>,
) -> Vec<Error<'a>> {
    map2.filter_map(|span| map1.get(&span).map(|_| Redifinition(span, span)))
        .collect()
}
