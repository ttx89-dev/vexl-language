//! VEXL Parser - Transforming tokens into AST using chumsky

use chumsky::prelude::*;
use crate::ast::*;
use crate::lexer::Token;

/// Parse a VEXL source string into declarations
pub fn parse(source: &str) -> Result<Vec<Decl>, Vec<Simple<Token>>> {
    use logos::Logos;

    // Tokenize the source
    let tokens: Vec<Token> = Token::lexer(source)
        .map(|tok| tok.unwrap_or(Token::Semicolon)) // Error recovery
        .collect();

    // Parse the tokens into an AST
    program_parser().parse(tokens)
}

/// Parse a VEXL source string for LSP, converting errors to strings
pub fn parse_for_lsp(source: &str) -> Result<Vec<Decl>, String> {
    parse(source).map_err(|errors| {
        errors.iter()
            .map(|e| format!("Parse error: {:?}", e))
            .collect::<Vec<_>>()
            .join("\n")
    })
}

/// Backward compatibility: parse as expression if no declarations found
pub fn parse_expr(source: &str) -> Result<Expr, Vec<Simple<Token>>> {
    use logos::Logos;
    let tokens: Vec<Token> = Token::lexer(source)
        .map(|tok| tok.unwrap_or(Token::Semicolon))
        .collect();
    
    parser().parse(tokens)
}

/// Main expression parser with support for all VEXL expression types
fn parser() -> impl Parser<Token, Expr, Error = Simple<Token>> {
    let span = Span { start: 0, end: 0 }; // Placeholder span
    
    recursive(|expr| {
        // ===== LITERALS =====
        let int = select! { Token::Int(n) => Expr::Int(n, span) };
        let float = select! { Token::Float(f) => Expr::Float(f, span) };
        let bool_lit = select! {
            Token::True => Expr::Bool(true, span),
            Token::False => Expr::Bool(false, span)
        };
        let string = select! { Token::String(s) => Expr::String(s, span) };
        let ident = select! { Token::Ident(s) => Expr::Ident(s, span) };
        
        // ===== RANGES =====
        // Infinite range: [0..]
        let infinite_range = expr.clone()
            .then_ignore(just(Token::DotDot))
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(move |start| Expr::InfiniteRange(Box::new(start), span));
        
        // Range: [0..10]
        let range = expr.clone()
            .then_ignore(just(Token::DotDot))
            .then(expr.clone())
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(move |(start, end)| Expr::Range(Box::new(start), Box::new(end), span));
        
        
        // ===== COMPREHENSION =====
        // [x*2 | x <- xs, x > 0]
        // Must come BEFORE vector literal to avoid ambiguity
        let comprehension = expr.clone()
            .then_ignore(just(Token::Pipe))
            .then(
                select! { Token::Ident(s) => s }
                    .then_ignore(just(Token::LeftArrow))
                    .then(expr.clone())
                    .separated_by(just(Token::Comma))
                    .at_least(1)
            )
            .then(
                just(Token::Comma)
                    .ignore_then(expr.clone())
                    .or_not()
            )
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(move |((element, bindings), filter)| Expr::Comprehension {
                element: Box::new(element),
                bindings,
                filter: filter.map(Box::new),
                span,
            });
        
        // ===== VECTOR LITERAL =====
        let vector = expr.clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map(move |elements| Expr::Vector(elements, span));
        
        // ===== LAMBDA EXPRESSIONS =====
        // Pipe syntax: |x, y| expr
        let lambda_pipe = select! { Token::Ident(s) => s }
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::Pipe), just(Token::Pipe))
            .then(expr.clone())
            .map(move |(params, body)| Expr::Lambda {
                params,
                body: Box::new(body),
                span,
            });
        
        // Arrow syntax: (x, y) => expr
        let lambda_arrow = select! { Token::Ident(s) => s }
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .then_ignore(just(Token::FatArrow))
            .then(expr.clone())
            .map(move |(params, body)| Expr::Lambda {
                params,
                body: Box::new(body),
                span,
            });
        
        let lambda = lambda_pipe.or(lambda_arrow);
        
        // ===== LET BINDING =====
        // let x = value in body
        let let_expr = just(Token::Let)
            .ignore_then(select! { Token::Ident(s) => s })
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then(expr.clone())
            .map(move |((name, value), body)| Expr::Let {
                name,
                type_annotation: None,
                value: Box::new(value),
                body: Box::new(body),
                span,
            });
        
        // ===== IF EXPRESSION =====
        // if cond then_expr else else_expr
        let if_expr = just(Token::If)
            .ignore_then(expr.clone())
            .then(expr.clone())
            .then(just(Token::Else).ignore_then(expr.clone()).or_not())
            .map(move |((cond, then_branch), else_branch)| Expr::If {
                cond: Box::new(cond),
                then_branch: Box::new(then_branch),
                else_branch: else_branch.map(Box::new),
                span,
            });
        
        // ===== FIX (RECURSION) =====
        // fix f => expr
        let fix_expr = just(Token::Fix)
            .ignore_then(select! { Token::Ident(s) => s })
            .then_ignore(just(Token::FatArrow))
            .then(expr.clone())
            .map(move |(name, body)| Expr::Fix {
                name,
                body: Box::new(body),
                span,
            });
        
        // ===== PARENTHESIZED EXPRESSION =====
        let paren = expr.clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));
        
        // ===== PRIMARY EXPRESSIONS =====
        let atom = int
            .or(float)
            .or(bool_lit)
            .or(string)
            .or(let_expr)
            .or(if_expr)
            .or(fix_expr)
            .or(lambda)
            .or(comprehension)   // Must come before vector/range
            .or(infinite_range)  // Must come before range
            .or(range)
            .or(vector)
            .or(ident)
            .or(paren);
        
        // ===== FUNCTION APPLICATION =====
        // f(x, y, z)
        let app = atom.clone()
            .then(
                expr.clone()
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .delimited_by(just(Token::LParen), just(Token::RParen))
                    .repeated()
            )
            .foldl(move |func, args| Expr::App {
                func: Box::new(func),
                args,
                span,
            });
        
        // ===== MATRIX OPERATIONS =====
        // @, **, *.
        let matrix_ops = app.clone()
            .then(
                choice((
                    just(Token::At).to(BinOpKind::MatMul),
                    just(Token::StarStar).to(BinOpKind::Outer),
                    just(Token::StarDot).to(BinOpKind::Dot),
                ))
                .then(app.clone())
                .repeated()
            )
            .foldl(move |left, (op, right)| Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            });
        
        // ===== MULTIPLICATION / DIVISION =====
        let product = matrix_ops.clone()
            .then(
                choice((
                    just(Token::Star).to(BinOpKind::Mul),
                    just(Token::Slash).to(BinOpKind::Div),
                ))
                .then(matrix_ops.clone())
                .repeated()
            )
            .foldl(move |left, (op, right)| Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            });
        
        // ===== ADDITION / SUBTRACTION =====
        let sum = product.clone()
            .then(
                choice((
                    just(Token::Plus).to(BinOpKind::Add),
                    just(Token::Minus).to(BinOpKind::Sub),
                ))
                .then(product.clone())
                .repeated()
            )
            .foldl(move |left, (op, right)| Expr::BinOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
                span,
            });
        
        // ===== COMPARISON =====
        let comparison = sum.clone()
            .then(
                choice((
                    just(Token::EqEq).to(BinOpKind::Eq),
                    just(Token::NotEq).to(BinOpKind::NotEq),
                    just(Token::Le).to(BinOpKind::Le),
                    just(Token::Lt).to(BinOpKind::Lt),
                    just(Token::Ge).to(BinOpKind::Ge),
                    just(Token::Gt).to(BinOpKind::Gt),
                ))
                .then(sum.clone())
                .or_not()
            )
            .map(move |(left, rest)| {
                if let Some((op, right)) = rest {
                    Expr::BinOp {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                        span,
                    }
                } else {
                    left
                }
            });
        
        // ===== PIPELINE =====
        let pipeline = comparison.clone()
            .separated_by(just(Token::Pipeline))
            .at_least(1)
            .map(move |stages| {
                if stages.len() == 1 {
                    stages.into_iter().next().unwrap()
                } else {
                    Expr::Pipeline { stages, span }
                }
            });
        
        pipeline
    })
}

/// Program parser that handles top-level declarations
fn program_parser() -> impl Parser<Token, Vec<Decl>, Error = Simple<Token>> {
    let span = Span { start: 0, end: 0 };

    // Type parser for function signatures
    let type_parser = recursive(|ty| {
        choice((
            select! { Token::Ident(s) => match s.as_str() {
                "i64" => Type::Int,
                "f64" => Type::Float,
                "bool" => Type::Bool,
                "string" => Type::String,
                other => Type::Named(other.to_string()),
            }},
            // Function type: (T1, T2) -> T3
            ty.clone()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .then_ignore(just(Token::Minus))
                .then_ignore(just(Token::Gt))
                .then(ty.clone())
                .map(|(params, ret)| Type::Function {
                    params,
                    ret: Box::new(ret),
                }),
        ))
    });

    // Function declaration: fn name(param: Type, ...) -> ReturnType { body }
    let function_decl = just(Token::Fn)
        .ignore_then(select! { Token::Ident(name) => name })
        .then(
            select! { Token::Ident(param_name) => param_name }
                .then_ignore(just(Token::Colon))
                .then(type_parser.clone())
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .delimited_by(just(Token::LParen), just(Token::RParen))
        )
        .then_ignore(just(Token::Minus))
        .then_ignore(just(Token::Gt))
        .then(type_parser)
        .then(parser().delimited_by(just(Token::LBrace), just(Token::RBrace)))
        .map(move |(((name, params), return_type), body)| Decl::Function {
            name,
            params,
            return_type,
            body,
            span,
        });

    // Expression as declaration
    let expr_decl = parser().map(move |expr| Decl::Expr(expr));

    // Program is a sequence of declarations
    function_decl
        .or(expr_decl)
        .repeated()
        .at_least(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_int() {
        let result = parse_expr("42");
        assert!(result.is_ok());
        if let Ok(Expr::Int(n, _)) = result {
            assert_eq!(n, 42);
        } else {
            panic!("Expected Int expression");
        }
    }
    
    #[test]
    fn test_parse_float() {
        let result = parse_expr("3.14");
        assert!(result.is_ok());
        if let Ok(Expr::Float(f, _)) = result {
            assert!((f - 3.14).abs() < 0.001);
        } else {
            panic!("Expected Float expression");
        }
    }

    #[test]
    fn test_parse_string() {
        let result = parse_expr("\"hello\"");
        assert!(result.is_ok());
        if let Ok(Expr::String(s, _)) = result {
            assert_eq!(s, "hello");
        } else {
            panic!("Expected String expression");
        }
    }

    #[test]
    fn test_parse_bool_true() {
        let result = parse_expr("true");
        assert!(result.is_ok());
        if let Ok(Expr::Bool(b, _)) = result {
            assert_eq!(b, true);
        } else {
            panic!("Expected Bool expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_bool_false() {
        let result = parse_expr("false");
        assert!(result.is_ok());
        if let Ok(Expr::Bool(b, _)) = result {
            assert_eq!(b, false);
        } else {
            panic!("Expected Bool expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_ident() {
        let result = parse_expr("foo");
        assert!(result.is_ok());
        if let Ok(Expr::Ident(s, _)) = result {
            assert_eq!(s, "foo");
        } else {
            panic!("Expected Ident expression");
        }
    }

    #[test]
    fn test_parse_vector() {
        let result = parse_expr("[1, 2, 3]");
        assert!(result.is_ok());
        if let Ok(Expr::Vector(elements, _)) = result {
            assert_eq!(elements.len(), 3);
        } else {
            panic!("Expected Vector expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_binary_op() {
        let result = parse_expr("1 + 2");
        assert!(result.is_ok());
        if let Ok(Expr::BinOp { op, .. }) = result {
            assert_eq!(op, BinOpKind::Add);
        } else {
            panic!("Expected BinOp expression");
        }
    }

    #[test]
    fn test_parse_multiplication() {
        let result = parse_expr("3 * 4");
        assert!(result.is_ok());
        if let Ok(Expr::BinOp { op, .. }) = result {
            assert_eq!(op, BinOpKind::Mul);
        } else {
            panic!("Expected BinOp expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_comparison() {
        let result = parse_expr("x == 5");
        assert!(result.is_ok());
        if let Ok(Expr::BinOp { op, .. }) = result {
            assert_eq!(op, BinOpKind::Eq);
        } else {
            panic!("Expected BinOp expression");
        }
    }

    #[test]
    fn test_parse_pipeline() {
        let result = parse_expr("data |> map |> filter");
        assert!(result.is_ok());
        if let Ok(Expr::Pipeline { stages, .. }) = result {
            assert_eq!(stages.len(), 3);
        } else {
            panic!("Expected Pipeline expression");
        }
    }

    // ===== NEW TESTS =====

    #[test]
    fn test_parse_let_binding() {
        let result = parse_expr("let x = 5 in x");
        println!("Let binding result: {:?}", result);
        assert!(result.is_ok());
        if let Ok(Expr::Let { name, .. }) = result {
            assert_eq!(name, "x");
        } else {
            panic!("Expected Let expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_lambda_pipe() {
        let result = parse_expr("|x| x + 1");
        assert!(result.is_ok());
        if let Ok(Expr::Lambda { params, .. }) = result {
            assert_eq!(params.len(), 1);
            assert_eq!(params[0], "x");
        } else {
            panic!("Expected Lambda expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_lambda_arrow() {
        let result = parse_expr("(x, y) => x + y");
        assert!(result.is_ok());
        if let Ok(Expr::Lambda { params, .. }) = result {
            assert_eq!(params.len(), 2);
            assert_eq!(params[0], "x");
            assert_eq!(params[1], "y");
        } else {
            panic!("Expected Lambda expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_if_with_else() {
        let result = parse_expr("if x y else z");
        assert!(result.is_ok());
        if let Ok(Expr::If { else_branch, .. }) = result {
            assert!(else_branch.is_some());
        } else {
            panic!("Expected If expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_if_without_else() {
        let result = parse_expr("if cond then_expr");
        assert!(result.is_ok());
        matches!(result.unwrap(), Expr::If { .. });
    }

    #[test]
    fn test_parse_range() {
        let result = parse_expr("[0..10]");
        assert!(result.is_ok());
        matches!(result.unwrap(), Expr::Range(_, _, _));
    }

    #[test]
    fn test_parse_infinite_range() {
        let result = parse_expr("[5..]");
        assert!(result.is_ok());
        matches!(result.unwrap(), Expr::InfiniteRange(_, _));
    }

    #[test]
    fn test_parse_fix() {
        let result = parse_expr("fix f => f");
        assert!(result.is_ok());
        if let Ok(Expr::Fix { name, .. }) = result {
            assert_eq!(name, "f");
        } else {
            panic!("Expected Fix expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_function_application() {
        let result = parse_expr("f(x, y)");
        assert!(result.is_ok());
        if let Ok(Expr::App { args, .. }) = result {
            assert_eq!(args.len(), 2);
        } else {
            panic!("Expected App expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_matrix_multiply() {
        let result = parse_expr("a @ b");
        assert!(result.is_ok());
        if let Ok(Expr::BinOp { op, .. }) = result {
            assert_eq!(op, BinOpKind::MatMul);
        } else {
            panic!("Expected BinOp expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_comprehension() {
        let result = parse_expr("[x | x <- xs]");
        assert!(result.is_ok());
        if let Ok(Expr::Comprehension { bindings, .. }) = result {
            assert_eq!(bindings.len(), 1);
            assert_eq!(bindings[0].0, "x");
        } else {
            panic!("Expected Comprehension expression but got {:?}", result);
        }
    }

    #[test]
    fn test_parse_comprehension_with_filter() {
        let result = parse_expr("[x * 2 | x <- nums, x > 0]");
        assert!(result.is_ok());
        if let Ok(Expr::Comprehension { bindings, filter, .. }) = result {
            assert_eq!(bindings.len(), 1);
            assert!(filter.is_some());
        } else {
            panic!("Expected Comprehension with filter but got {:?}", result);
        }
    }
}
