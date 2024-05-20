use crate::{
    ast::{FunctionSig, SigSymMap, Var, VarSymMap},
    cst::{Import, PVar},
    hir::error::Error::{self, *},
    span::*,
};

use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Copy)]
pub struct SymTable<'a, O> {
    map: &'a HashMap<String, O>,
    parent: Option<&'a Self>,
}

pub type FSymMap<'a> = SymTable<'a, FunctionSig>;
pub type VSymMap<'a> = SymTable<'a, Var>;

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
    externs: &[Import<'a>],
) -> Result<SigSymMap, Vec<Error<'a>>> {
    let mut errors = vec![];
    for i in 0..externs.len() {
        for j in 0..i {
            if externs[i].name() == externs[j].name() {
                errors.push(Redifinition(externs[i].name(), externs[j].name()));
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
    vars.as_ref()
        .iter()
        .fold(Ok(HashMap::new()), |r, v| match (r, Var::from_pvar(*v)) {
            (Ok(mut syms), Ok(sym)) => {
                syms.insert(sym.name().to_string(), sym);
                Ok(syms)
            }
            (Err(mut errors), Err(e)) => {
                errors.push(e);
                Err(errors)
            }
            (Ok(_), Err(e)) => Err(vec![e]),
            (errors @ Err(_), Ok(_)) => errors,
        })
}

pub(super) fn get_redefs<'a>(syms: impl Iterator<Item = Span<'a>>) -> Option<Vec<Error<'a>>> {
    let redefs = syms
        .fold((vec![], HashSet::new()), |(mut redefs, mut syms), sym| {
            if let Some(old_f) = syms.get(&sym) {
                redefs.push(Error::Redifinition(*old_f, sym));
            } else {
                syms.insert(sym);
            }
            (redefs, syms)
        })
        .0;
    if redefs.is_empty() {
        None
    } else {
        Some(redefs)
    }
}
