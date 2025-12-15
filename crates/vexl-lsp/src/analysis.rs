
use dashmap::DashMap;
use tower_lsp::lsp_types::Url;

#[derive(Debug)]
pub struct Analysis {
    // Store file contents: URI -> Content
    pub documents: DashMap<Url, String>,
}

impl Analysis {
    pub fn new() -> Self {
        Self {
            documents: DashMap::new(),
        }
    }

    pub fn update_file(&self, uri: Url, text: String) {
        self.documents.insert(uri, text);
    }

    #[allow(dead_code)]
    pub fn get_file<'a>(&'a self, uri: &Url) -> Option<dashmap::mapref::one::Ref<'a, Url, String>> {
        self.documents.get(uri)
    }
}
