//! VEXL Parser - Simple manual parser for basic functionality

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

    // Parse the tokens into an AST using simple manual parser
    simple_parse(&tokens)
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
    use chumsky::prelude::*;
    let span = Span { start: 0, end: 0 }; // Placeholder span

    // Simple binary expression parser for testing
    let int_lit = select! { Token::Int(n) => Expr::Int(n, span) };
    let string_lit = select! { Token::String(s) => Expr::String(s, span) };
    let ident = select! { Token::Ident(s) => Expr::Ident(s, span) };

    let primary = int_lit.or(string_lit).or(ident);

    // Simple addition for testing
    primary
        .clone()
        .then(just(Token::Plus).to(BinOpKind::Add).then(primary).repeated())
        .foldl(move |left, (op, right)| Expr::BinOp {
            op,
            left: Box::new(left),
            right: Box::new(right),
            span,
        })
}

/// Simple manual parser for basic functionality
fn simple_parse(tokens: &[Token]) -> Result<Vec<Decl>, Vec<Simple<Token>>> {
    let mut declarations = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        match &tokens[i] {
            Token::Let => {
                // Parse let binding: let name = expr
                i += 1; // skip 'let'
                if i >= tokens.len() {
                    eprintln!("DEBUG: Bounds check failed after 'let', i={}", i);
                    return Err(vec![]);
                }

                let name = match &tokens[i] {
                    Token::Ident(n) => n.clone(),
                    _ => {
                        eprintln!("DEBUG: Expected identifier after 'let', i={}, token={:?}", i, tokens.get(i));
                        return Err(vec![]);
                    }
                };
                i += 1; // skip name

                if i >= tokens.len() || tokens[i] != Token::Eq {
                    eprintln!("DEBUG: Expected '=' after name, i={}, token={:?}", i, tokens.get(i));
                    return Err(vec![]);
                }
                i += 1; // skip '='

                // Parse expression (support strings, ints, idents, concatenation, pipelines)
                // First try to parse a simple expression, then check for pipelines
                eprintln!("DEBUG: Before parse_expression, i={}, token={:?}", i, tokens.get(i));
                let mut value = parse_expression(tokens, &mut i)?;
                eprintln!("DEBUG: After parse_expression, i={}, token={:?}", i, tokens.get(i));

                // Check for pipeline operations
                while i < tokens.len() && tokens[i] == Token::Pipeline {
                    i += 1; // skip '|>'

                    // Parse the function call after pipeline
                    if i >= tokens.len() {
                        return Err(vec![]);
                    }

                    match &tokens[i] {
                        Token::Ident(func_name) => {
                            i += 1; // skip function name

                            if i >= tokens.len() || tokens[i] != Token::LParen {
                                return Err(vec![]);
                            }
                            i += 1; // skip '('

                            // Parse arguments - for now, handle simple lambda |x| expr
                            let mut args = Vec::new();

                            if i < tokens.len() && tokens[i] == Token::Pipe {
                                // Lambda function: |x| expr
                                i += 1; // skip '|'

                                // Parse parameter
                                let param = match &tokens[i] {
                                    Token::Ident(p) => p.clone(),
                                    _ => return Err(vec![]),
                                };
                                i += 1; // skip param

                                if i >= tokens.len() || tokens[i] != Token::Pipe {
                                    return Err(vec![]);
                                }
                                i += 1; // skip '|'

                                // Parse lambda body (simple expression for now)
                                let body = parse_expression(tokens, &mut i)?;

                                // Create lambda
                                let lambda = Expr::Lambda {
                                    params: vec![param],
                                    body: Box::new(body),
                                    span: Span { start: 0, end: 0 },
                                };
                                args.push(lambda);
                            } else {
                                // Regular arguments
                                while i < tokens.len() && tokens[i] != Token::RParen {
                                    let arg = parse_expression(tokens, &mut i)?;
                                    args.push(arg);

                                    // Skip comma if present
                                    if i < tokens.len() && tokens[i] == Token::Comma {
                                        i += 1;
                                    }
                                }
                            }

                            if i >= tokens.len() || tokens[i] != Token::RParen {
                                return Err(vec![]);
                            }
                            i += 1; // skip ')'

                            // Create pipeline: left |> right becomes right(left, args...)
                            // Insert the pipeline input as the first argument
                            let mut pipeline_args = vec![value];
                            pipeline_args.extend(args);

                            value = Expr::App {
                                func: Box::new(Expr::Ident(func_name.clone(), Span { start: 0, end: 0 })),
                                args: pipeline_args,
                                span: Span { start: 0, end: 0 },
                            };
                        }
                        _ => return Err(vec![]),
                    }
                }

                // Create a let expression
                let let_expr = Expr::Let {
                    name,
                    type_annotation: None,
                    value: Box::new(value),
                    body: Box::new(Expr::Int(0, Span { start: 0, end: 0 })), // dummy body
                    span: Span { start: 0, end: 0 },
                };
                declarations.push(Decl::Expr(let_expr));
            }
            Token::Ident(name) => {
                // Check if this is a function call: ident(args)
                let func_name = name.clone();
                i += 1; // skip the function name

                if i >= tokens.len() || tokens[i] != Token::LParen {
                    // Not a function call, treat as identifier expression
                    declarations.push(Decl::Expr(Expr::Ident(func_name, Span { start: 0, end: 0 })));
                } else {
                    // Function call
                    i += 1; // skip '('

                    // Parse arguments (can be multiple)
                    let mut args = Vec::new();
                    while i < tokens.len() && tokens[i] != Token::RParen {
                        let arg = parse_expression(tokens, &mut i)?;
                        args.push(arg);

                        // Skip comma if present
                        if i < tokens.len() && tokens[i] == Token::Comma {
                            i += 1;
                        }
                    }

                    if i >= tokens.len() || tokens[i] != Token::RParen {
                        return Err(vec![]);
                    }
                    i += 1; // skip ')'

                    declarations.push(Decl::Expr(Expr::App {
                        func: Box::new(Expr::Ident(func_name, Span { start: 0, end: 0 })),
                        args,
                        span: Span { start: 0, end: 0 },
                    }));
                }
            }
            Token::Int(n) => {
                // Simple integer expression
                let expr = Expr::Int(*n, Span { start: 0, end: 0 });
                declarations.push(Decl::Expr(expr));
                i += 1;
            }
            Token::String(s) => {
                // Simple string expression
                let expr = Expr::String(s.clone(), Span { start: 0, end: 0 });
                declarations.push(Decl::Expr(expr));
                i += 1;
            }
            Token::LBracket => {
                // Parse vector: [expr, expr, ...]
                i += 1; // skip '['
                let mut elements = Vec::new();

                while i < tokens.len() && tokens[i] != Token::RBracket {
                    let elem = parse_expression(tokens, &mut i)?;
                    elements.push(elem);

                    // Skip comma if present
                    if i < tokens.len() && tokens[i] == Token::Comma {
                        i += 1;
                    }
                }

                if i >= tokens.len() || tokens[i] != Token::RBracket {
                    return Err(vec![]);
                }
                i += 1; // skip ']'

                declarations.push(Decl::Expr(Expr::Vector(elements, Span { start: 0, end: 0 })));
            }

            token => {
                // Unknown token - this might indicate a parsing issue
                eprintln!("DEBUG: Unknown token at position {}: {:?}, remaining tokens: {:?}", i, token, &tokens[i..i.min(i+5)].to_vec());
                return Err(vec![]);
            }
        }
    }

    Ok(declarations)
}

/// Parse a single expression starting at the given index
fn parse_expression(tokens: &[Token], i: &mut usize) -> Result<Expr, Vec<Simple<Token>>> {
    // let start_i = *i; // Not currently used

    // Parse primary expression
    let mut expr = match &tokens[*i] {
        Token::Int(n) => {
            *i += 1;
            Expr::Int(*n, Span { start: 0, end: 0 })
        }
        Token::String(s) => {
            *i += 1;
            Expr::String(s.clone(), Span { start: 0, end: 0 })
        }
        Token::Ident(s) => {
            *i += 1;
            Expr::Ident(s.clone(), Span { start: 0, end: 0 })
        }
        Token::LBracket => {
            // Parse vector: [expr, expr, ...]
            *i += 1; // skip '['
            let mut elements = Vec::new();

            while *i < tokens.len() && tokens[*i] != Token::RBracket {
                let elem = parse_expression(tokens, i)?;
                elements.push(elem);

                // Skip comma if present
                if *i < tokens.len() && tokens[*i] == Token::Comma {
                    *i += 1;
                }
            }

            if *i >= tokens.len() || tokens[*i] != Token::RBracket {
                return Err(vec![]);
            }
            *i += 1; // skip ']'

            Expr::Vector(elements, Span { start: 0, end: 0 })
        }
        _ => {
            eprintln!("DEBUG: parse_expression failed at i={}, token={:?}", *i, tokens.get(*i));
            return Err(vec![]);
        }
    };

    // Check for binary operations (+ for concatenation, comparisons, etc.)
    // Use a loop to handle left-associative operations: a + b + c becomes (a + b) + c
    while *i < tokens.len() {
        let op = match &tokens[*i] {
            Token::Plus => BinOpKind::Add,
            Token::Minus => BinOpKind::Sub,
            Token::Star => BinOpKind::Mul,
            Token::Slash => BinOpKind::Div,
            Token::Gt => BinOpKind::Gt,
            Token::Lt => BinOpKind::Lt,
            Token::Ge => BinOpKind::Ge,
            Token::Le => BinOpKind::Le,
            Token::EqEq => BinOpKind::Eq,
            Token::NotEq => BinOpKind::NotEq,
            _ => break, // Not a binary operator
        };

        *i += 1; // skip operator

        // Parse right-hand side
        let right = match &tokens[*i] {
            Token::Int(n) => {
                *i += 1;
                Expr::Int(*n, Span { start: 0, end: 0 })
            }
            Token::String(s) => {
                *i += 1;
                Expr::String(s.clone(), Span { start: 0, end: 0 })
            }
            Token::Ident(s) => {
                *i += 1;
                Expr::Ident(s.clone(), Span { start: 0, end: 0 })
            }
            Token::LBracket => {
                // Parse vector on right side
                *i += 1; // skip '['
                let mut elements = Vec::new();

                while *i < tokens.len() && tokens[*i] != Token::RBracket {
                    let elem = parse_expression(tokens, i)?;
                    elements.push(elem);

                    if *i < tokens.len() && tokens[*i] == Token::Comma {
                        *i += 1;
                    }
                }

                if *i >= tokens.len() || tokens[*i] != Token::RBracket {
                    return Err(vec![]);
                }
                *i += 1; // skip ']'

                Expr::Vector(elements, Span { start: 0, end: 0 })
            }
            _ => return Err(vec![]),
        };

        // Create binary operation (left-associative)
        expr = Expr::BinOp {
            op,
            left: Box::new(expr),
            right: Box::new(right),
            span: Span { start: 0, end: 0 },
        };
    }

    Ok(expr)
}

/// Parse a pipeline expression: expr |> func(args)
fn parse_pipeline_expression(tokens: &[Token], i: &mut usize) -> Result<Expr, Vec<Simple<Token>>> {
    // Parse the base expression first
    let mut expr = parse_expression(tokens, i)?;

    // Check for pipeline operations
    while *i < tokens.len() && tokens[*i] == Token::Pipeline {
        *i += 1; // skip '|>'

        // Parse the function call after pipeline
        if *i >= tokens.len() {
            return Err(vec![]);
        }

        match &tokens[*i] {
            Token::Ident(func_name) => {
                *i += 1; // skip function name

                if *i >= tokens.len() || tokens[*i] != Token::LParen {
                    return Err(vec![]);
                }
                *i += 1; // skip '('

                // Parse arguments - for now, handle simple lambda |x| expr
                let mut args = Vec::new();

                if *i < tokens.len() && tokens[*i] == Token::Pipe {
                    // Lambda function: |x| expr
                    *i += 1; // skip '|'

                    // Parse parameter
                    let param = match &tokens[*i] {
                        Token::Ident(p) => p.clone(),
                        _ => return Err(vec![]),
                    };
                    *i += 1; // skip param

                    if *i >= tokens.len() || tokens[*i] != Token::Pipe {
                        return Err(vec![]);
                    }
                    *i += 1; // skip '|'

                    // Parse lambda body (simple expression for now)
                    let body = parse_expression(tokens, i)?;

                    // Create lambda
                    let lambda = Expr::Lambda {
                        params: vec![param],
                        body: Box::new(body),
                        span: Span { start: 0, end: 0 },
                    };
                    args.push(lambda);
                } else {
                    // Regular arguments
                    while *i < tokens.len() && tokens[*i] != Token::RParen {
                        let arg = parse_expression(tokens, i)?;
                        args.push(arg);

                        // Skip comma if present
                        if *i < tokens.len() && tokens[*i] == Token::Comma {
                            *i += 1;
                        }
                    }
                }

                if *i >= tokens.len() || tokens[*i] != Token::RParen {
                    return Err(vec![]);
                }
                *i += 1; // skip ')'

                // Create function application
                let func_call = Expr::App {
                    func: Box::new(Expr::Ident(func_name.clone(), Span { start: 0, end: 0 })),
                    args,
                    span: Span { start: 0, end: 0 },
                };

                // Create pipeline: left |> right becomes right(left)
                expr = Expr::App {
                    func: Box::new(func_call),
                    args: vec![expr],
                    span: Span { start: 0, end: 0 },
                };
            }
            _ => return Err(vec![]),
        }
    }

    Ok(expr)
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
    fn test_parse_vector() {
        let result = parse("[1, 2, 3]");
        assert!(result.is_ok());
        if let Ok(decls) = result {
            assert_eq!(decls.len(), 1);
            if let Decl::Expr(Expr::Vector(elements, _)) = &decls[0] {
                assert_eq!(elements.len(), 3);
            } else {
                panic!("Expected Vector expression");
            }
        }
    }
}
