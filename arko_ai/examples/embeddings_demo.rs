use arko_ai::{
    EmbeddingClient, EmbeddingConfig, EmbeddingProviderIdentifier, Error, VoyageInputType,
};

#[tokio::main]
async fn main() -> Result<(), Error> {
    dotenv::dotenv().ok();
    let voyage_api_key = std::env::var("VOYAGE_API_KEY").unwrap_or_default();

    let client = EmbeddingClient::new().with_voyage_api_key(&voyage_api_key);

    let texts_to_embed = vec![
        "Hello world!".to_string(),
        "This is a test of the embedding system.".to_string(),
    ];

    if !voyage_api_key.is_empty() {
        println!("\n--- Testing Voyage Embedding ---");
        let voyage_config =
            EmbeddingConfig::new(EmbeddingProviderIdentifier::Voyage, "voyage-4-lite")
                .with_voyage_input_type(VoyageInputType::Document)
                .with_output_dimensionality(256);

        let first_text_tokens = client.count_tokens(&texts_to_embed[0], &voyage_config)?;
        println!(
            "Tokens for '{}' (Voyage): {}",
            texts_to_embed[0], first_text_tokens
        );

        let estimated_cost_voyage = client.get_embedding_cost(&texts_to_embed, &voyage_config)?;
        println!(
            "Estimated cost for Voyage embedding: ${:.9}",
            estimated_cost_voyage
        );

        match client.embed(texts_to_embed.clone(), &voyage_config).await {
            Ok(response) => {
                println!(
                    "Voyage Embeddings received: {} vectors.",
                    response.embeddings.len()
                );
                if let Some(first_embedding) = response.embeddings.first() {
                    println!(
                        "First Voyage embedding vector (first 5 dims): {:?}",
                        first_embedding.iter().take(5).collect::<Vec<_>>()
                    );
                }
                println!("Voyage Usage Info: {:?}", response.usage);
            }
            Err(e) => eprintln!("Voyage embedding error: {}", e),
        }
    } else {
        println!("\nSkipping Voyage Embedding Test: VOYAGE_API_KEY not set.");
    }

    Ok(())
}
