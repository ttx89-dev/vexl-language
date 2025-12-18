
use tower_lsp::jsonrpc::Result;
use tower_lsp::lsp_types::*;
use tower_lsp::{Client, LanguageServer};

use crate::analysis::Analysis;
use crate::features::{completion, hover, diagnostics};

#[derive(Debug)]
pub struct Backend {
    client: Client,
    #[allow(dead_code)] // For now
    analysis: Analysis,
}

impl Backend {
    pub fn new(client: Client) -> Self {
        Self { 
            client,
            analysis: Analysis::new(),
        }
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                completion_provider: Some(CompletionOptions::default()),
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "VEXL LSP initialized!")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn completion(&self, params: CompletionParams) -> Result<Option<CompletionResponse>> {
        let _ = params;
        // In a real implementation we would look up the text from our state
        // For now, passing empty string to stub
        Ok(completion::complete("", params.text_document_position.position))
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        Ok(hover::hover("", params.text_document_position_params.position))
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        self.client
            .log_message(MessageType::INFO, format!("opened document: {}", params.text_document.uri))
            .await;
        
        self.analysis.update_file(params.text_document.uri.clone(), params.text_document.text.clone());

        let diags = diagnostics::diagnostics(&params.text_document.text, "current").unwrap_or_default();
        self.client.publish_diagnostics(params.text_document.uri, diags, None).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        // Since we asked for FULL sync, the first change event has the full text
        if let Some(change) = params.content_changes.first() {
             self.analysis.update_file(params.text_document.uri.clone(), change.text.clone());
             
             let diags = diagnostics::diagnostics(&change.text, "current").unwrap_or_default();
             self.client.publish_diagnostics(params.text_document.uri, diags, None).await;
        }
    }
}
