from src.models.retrieval_model import RetrievalDocument, RetrievalRequest
from src.retrieval import RetrievalSessionFactory
from src.retrieval.embeddings.retrieval_embedding_service import RetrievalEmbeddingService


def simple_token_counter(text: str) -> int:
    """Provides an approximate LLM token count for the standalone retrieval test."""

    return max(1, len(text) // 4) if text else 0


def main() -> None:
    """Runs a simple end-to-end retrieval test."""

    # Load the local embedding model once, just as the application will do at startup.
    embedding_service = RetrievalEmbeddingService(
        model_name="BAAI/bge-small-en-v1.5",
        device="cpu",
    )

    factory = RetrievalSessionFactory(
        embedding_service=embedding_service,
        context_token_counter=simple_token_counter,
        chunk_size=400,
        candidate_k=6,
        top_k=6,
        semantic_weight=0.5,
        lexical_weight=0.5,
        max_context_tokens=1500,
        max_chunks_per_document=2,
    )

    documents = [
        RetrievalDocument(
            document_id="acme-security",
            source_type="web",
            title="Acme Enterprise Security",
            content="""
Acme Enterprise supports SAML-based single sign-on for enterprise customers.
Administrators can integrate their existing identity provider and centrally
control employee authentication.

The platform also supports SCIM provisioning. User accounts can be automatically
created, updated, and deactivated when changes are made in the organization's
identity management system.

Customer data is encrypted at rest using AES-256 encryption. Data transmitted
between clients and Acme services is protected using TLS.
""",
            metadata={"url": "https://example.com/acme-security"},
        ),
        RetrievalDocument(
            document_id="acme-pricing",
            source_type="web",
            title="Acme Enterprise Pricing",
            content="""
Acme offers multiple subscription plans for businesses.

Enterprise pricing is negotiated based on employee count, storage requirements,
support requirements, and contract duration. Volume discounts may be available
for large organizations.

The enterprise plan includes priority customer support and a dedicated account
manager.
""",
            metadata={"url": "https://example.com/acme-pricing"},
        ),
        RetrievalDocument(
            document_id="contoso-security",
            source_type="web",
            title="Contoso Security",
            content="""
Contoso provides multi-factor authentication and role-based access control.

Enterprise customers can configure authentication policies and restrict access
based on organization security requirements.
""",
            metadata={"url": "https://example.com/contoso-security"},
        ),
    ]

    session = factory.create_session()

    try:
        # Chunk and index the corpus only once for the entire research session.
        session.build(documents)

        request = RetrievalRequest(
            subquestion_id="sq-1",
            research_question="Compare Acme and Contoso for enterprise readiness.",
            subquestion="What identity management and authentication features does Acme Enterprise provide?",
            search_queries=[
                "Acme Enterprise SAML SSO",
                "Acme SCIM provisioning",
            ],
        )

        result = session.retrieve(request)

        print("\nRetrieval query:")
        print(result.retrieval_query)

        print("\nRetrieved chunks:")

        for retrieved_chunk in result.chunks:
            chunk = retrieved_chunk.chunk

            print(
                f"\nRank {retrieved_chunk.rank}"
                f"\nDocument: {chunk.document_id}"
                f"\nTitle: {chunk.title}"
                f"\nContent: {chunk.content}"
            )

        print("\nFinal context:")
        print(result.context)

        print(f"\nEstimated context tokens: {result.estimated_tokens}")

    finally:
        # Remove all job-specific chunks and retrieval indexes after the test completes.
        session.clear()


if __name__ == "__main__":
    main()

