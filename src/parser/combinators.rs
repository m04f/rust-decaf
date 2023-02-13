use crate::span::Spanned;

/// the parser trait assumes that error handling is done by the parser and the error it self does
/// not get returned by the parser although it's side effects can reflect in the returned
/// `ParseResult`
trait Parser<'a, I, O, E> {
    fn parse(&mut self, input: I, error_handler: impl FnMut(Spanned<'a, E>))
        -> (ParseResult<O>, I);
}

impl<'a, I, O, E: Clone, F> Parser<'a, I, O, E> for F
where
    F: FnMut(I) -> (ParseResult<O>, I),
{
    fn parse(
        &mut self,
        input: I,
        error_handler: impl FnMut(Spanned<'a, E>),
    ) -> (ParseResult<O>, I) {
        (*self)(input, error_handler)
    }
}

trait ErrorHandler<'a, T> {
    fn handle(&mut self, expected: Spanned<'a, T>);
}

enum ParseResult<O> {
    /// we could parse what we wanted without any errors
    Parsed(O),
    /// some handling have occured, but we did not necessarly parse something
    Recovered(Option<O>),
    /// no parsing have happened at all. (this means that the input did not change).
    Nil,
}

impl<'a, T, F> ErrorHandler<'a, T> for F
where
    F: FnMut(Spanned<'a, T>),
{
    fn handle(&mut self, expected: Spanned<'a, T>) {
        (*self)(expected)
    }
}
//
// fn and<'a, I, O1, O2, E, EH: FnMut(Spanned<'a, E>)>(
//     mut p1: impl Parser<'a, I, O1, E, EH>,
//     mut p2: impl Parser<'a, I, O2, E, EH>,
//     mut error_handler: EH,
// ) -> impl FnMut(I) -> (ParseResult<(O1, O2)>, I) {
//     move |input| {
//         let (o1, rem) = p1.parse(input, &mut error_handler);
//         match o1 {
//             ParseResult::Nil => (ParseResult::Nil, rem),
//             ParseResult::Recovered(None) => (ParseResult::Recovered(None), rem),
//             ParseResult::Parsed(o1) => match p2.parse(rem, &mut error_handler) {
//                 (ParseResult::Nil, rem) => (ParseResult::Recovered(None), rem),
//                 (ParseResult::Recovered(None), rem) => (ParseResult::Recovered(None), rem),
//                 (ParseResult::Recovered(Some(o2)), rem) => {
//                     (ParseResult::Recovered(Some((o1, o2))), rem)
//                 }
//                 (ParseResult::Parsed(o2), rem) => (ParseResult::Parsed((o1, o2)), rem),
//             },
//             ParseResult::Recovered(Some(o1)) => match p2.parse(rem, &mut error_handler) {
//                 (ParseResult::Nil, rem) => (ParseResult::Recovered(None), rem),
//                 (ParseResult::Recovered(None), rem) => (ParseResult::Recovered(None), rem),
//                 (ParseResult::Recovered(Some(o2)), rem) => {
//                     (ParseResult::Recovered(Some((o1, o2))), rem)
//                 }
//                 (ParseResult::Parsed(o2), rem) => (ParseResult::Recovered(Some((o1, o2))), rem),
//             },
//         }
//     }
// }
// //
// // // /// converts the parser's output to a spanned output.
// // // fn recognize<'a, O: Clone, Tokens: TokenStream<'a>>(
// // //     mut parser: impl Parser<'a, Tokens, O>,
// // // ) -> impl FnMut(Tokens) -> (ParseResult<Spanned<'a, O>>, Tokens) {
// // //     move |tokens: Tokens| {
// // //         map(
// // //             and(position, and(|i| parser.parse(i), position)),
// // //             |(beg, (output, end))| beg.merge(end).into_spanned(output),
// // //         )(tokens)
// // //     }
// // // }
// //
// // fn map<'a, F, T, I, E, EH: ErrorHandler<'a, E>>(
// //     mut parser: impl Parser<'a, I, F, E, EH>,
// //     mut f: impl FnMut(F) -> T,
// //     mut error_handler: impl FnMut(Spanned<'a, E>),
// // ) -> impl FnMut(I) -> (ParseResult<T>, I) {
// //     move |input| {
// //         let (output, rem) = parser.parse(input, &mut error_handler);
// //         (output.map(|o| f(o)), rem)
// //     }
// // }
// //
// // fn opt<'a, I, O, E, EH: ErrorHandler<'a, E>>(
// //     mut p1: impl Parser<'a, I, O, E, EH>,
// //     error_handler: EH,
// // ) -> impl FnMut(I) -> (ParseResult<Option<O>>, I) {
// //     move |input| match p1.parse(input, error_handler) {
// //         (ParseResult::Nil, rem) => (ParseResult::Parsed(None), rem),
// //         (ParseResult::Recovered(None), rem) => (ParseResult::Recovered(None), rem),
// //         (ParseResult::Parsed(o), rem) => (ParseResult::Parsed(Some(o)), rem),
// //         (ParseResult::Recovered(Some(o)), rem) => (ParseResult::Recovered(Some(Some(o))), rem),
// //     }
// // }
// //
// // // fn sequence<'a, I: TokenStream<'a>, O>(
// // //     mut p: impl Parser<'a, I, O>,
// // //     mut separator: Token,
// // // ) -> impl FnMut(I) -> (ParseResult<Vec<O>>, I) {
// // //     move |mut input| {
// // //         let mut parsed = vec![];
// // //         loop {
// // //             match p.parse(input) {
// // //                 (ParseResult::Nil, rem) => return (ParseResult::Parsed(parsed), rem),
// // //                 // maybe we should not return here.
// // //                 (ParseResult::Recovered(None), rem) => return (ParseResult::Recovered(None), rem),
// // //                 (ParseResult::Parsed(o), rem) => {
// // //                     input = rem;
// // //                     parsed.push(o);
// // //                 }
// // //                 (ParseResult::Recovered(Some(o)), rem) => {
// // //                     input = rem;
// // //                     parsed.push(o);
// // //                 }
// // //             }
// // //             match separator.parse(input) {
// // //                 (tok, rem) if tok.is_nil() => return (ParseResult::Parsed(parsed), rem),
// // //                 (_, rem) => {
// // //                     input = rem;
// // //                 }
// // //             }
// // //         }
// // //     }
// // // }
// //
// // fn or<'a, I, O, E, EH: ErrorHandler<'a, E>>(
// //     mut p1: impl Parser<'a, I, O, E, EH>,
// //     mut p2: impl Parser<'a, I, O, E, EH>,
// //     mut error_handler: impl FnMut(Spanned<'a, E>),
// // ) -> impl FnMut(I) -> (ParseResult<O>, I) {
// //     move |input| {
// //         let (o1, rem) = p1.parse(input, error_handler);
// //         if o1.is_nil() {
// //             p2.parse(rem, error_handler)
// //         } else {
// //             (o1, rem)
// //         }
// //     }
// // }
