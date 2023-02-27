# Semantics project

## Convention

`P<type>`
:means *Parsed*-type

`HIR<type>`
:means *HIR*-type

## Design

* Instead of first constructing a HIR-AST then check it for errors, I borrowed
  from rust's `str` type the idea that a HIR-AST is always a valid AST. and the
  checks are made at the conversion from the `Parse-AST` to `HIR-AST`
  propagating any errors in the conversion. I personally like this design over
  first construct a HIR-AST then checking it because
    1. I am writing the project in rust and it is conventional to avoid
       constructing objects in invalid state.
    1. I would like to assume that any node in the AST is always a
       valid node.

* The conversion from `P<Node>` to `HIR<Node>` is done by
  `fn HIR<Node>::from_p<node>(node) -> Result<HIR<Node>, Vec<Error>>`. Maybe
  defining a trait for the conversion would have been good but I started using
  this fixed function signature for the conversion and it stuck.

* The errors returned when converting any node in the AST is implemented as
  `Vec<Error>`. Using a vector for errors allows me to report as much errors as
  possible. I have tried to use iterators instead of vectors but it turned out
  to be very hard to get it to work. `Generator` type was also interesting but
  it is still nightly and it was not easy to use. 

## Extras

TODO: write about the App trait

## Difficulties

* The lexer and the parser were designed to report errors in form of iterators.
  Which was very complex and I could not maintain that design and could not
  rewrite the parser. Which makes the design heterogeneous at the application
  level.
