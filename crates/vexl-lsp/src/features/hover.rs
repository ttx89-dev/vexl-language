
use tower_lsp::lsp_types::*;

pub fn hover(_text: &str, _position: Position) -> Option<Hover> {
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: "**VEXL Expression**\n\nType: `Vector<T, D>`".to_string(),
        }),
        range: None,
    })
}
