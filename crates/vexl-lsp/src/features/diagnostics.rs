
use tower_lsp::lsp_types::*;

pub fn get_diagnostics(text: &str) -> Vec<Diagnostic> {
    let mut diags = Vec::new();
    
    for (i, line) in text.lines().enumerate() {
        if let Some(col) = line.find("TODO") {
            diags.push(Diagnostic {
                range: Range {
                    start: Position {
                        line: i as u32,
                        character: col as u32,
                    },
                    end: Position {
                        line: i as u32,
                        character: (col + 4) as u32,
                    },
                },
                severity: Some(DiagnosticSeverity::WARNING),
                code: None,
                code_description: None,
                source: Some("vexl-lsp".to_string()),
                message: "Found TODO item".to_string(),
                related_information: None,
                tags: None,
                data: None,
            });
        }
    }
    
    diags
}
