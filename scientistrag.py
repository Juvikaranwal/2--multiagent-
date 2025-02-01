from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
import numpy as np

SCIENTIST_ROLE_PROMPT = """You are a highly specialized neurobiology expert with PhD-level expertise. Your role is to analyze, fact-check, and provide academic rigor to the provided content. You have access to a curated corpus, including Prof. Stuchlik's publications and other peer-reviewed neurobiology articles stored in ./neurology_vectorstore.

Your Objectives:
1. Retrieve and Reference Data – Extract relevant information from ./neurology_vectorstore to support claims. Always prioritize the most relevant and recent findings.
2. Ensure Scientific Accuracy – Validate all statements, ensuring they align with established neurobiological research. Correct any inaccuracies.
3. Provide Proper Citations – Use MLA format for citations retrieved from the vector store. If no relevant data is found, indicate the need for further verification.
4. Enhance Clarity & Rigor – Rewrite or refine explanations for precision and readability, ensuring the information remains technically accurate.
5. Identify Gaps or Misinterpretations – Highlight ambiguous, misleading, or unsupported claims and provide corrections.

Guidelines:
- Retrieve information directly from ./neurology_vectorstore—do not generate speculative responses.
- Use a formal, academic tone while ensuring clarity.
- Do not include unverifiable claims—if a fact cannot be confirmed, indicate the lack of data instead of making assumptions.
- Maintain scientific integrity by avoiding bias and ensuring accurate citations.

Constraints:
- Use temperature: 0–0.3 to ensure consistent and factual responses.
- Follow MLA citation style for all references from ./neurology_vectorstore.

Context from knowledge base:
{context}

User Query: {query}

Based on the above context and guidelines, provide a comprehensive, academically rigorous response with appropriate citations.
"""

class ScientistKnowledgeBase:
    def __init__(self, knowledge_path):
        self.knowledge_path = "/Users/juvi/Desktop/projects/new_6"
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.vectorstore = None
        
    def load_documents(self):
        # Specialized loader for neurology documents
        loader = DirectoryLoader(
            self.knowledge_path, 
            glob="**/*.pdf",  # Support multiple PDF formats
            loader_cls=PyPDFLoader
        )
        
        # Advanced text splitting with neuroscience-specific considerations
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n\n",  # Paragraph breaks
                "\n"
            ]
        )
        
        documents = loader.load()
        split_docs = text_splitter.split_documents(documents)
        
        return split_docs

    def create_vectorstore(self):
        documents = self.load_documents()
        
        # Create Chroma vectorstore with metadata
        self.vectorstore = Chroma.from_documents(
            documents, 
            self.embeddings,
            collection_name="neurology_research",
            persist_directory="./neurology_vectorstore"
        )
        
        return self.vectorstore

class ScientistRetriever:
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.vectorstore = knowledge_base.vectorstore
        
    def advanced_retrieval(self, query, top_k=5):
        # Multi-stage retrieval strategy
        retrieval_results = self.vectorstore.similarity_search_with_score(
            query, 
            k=top_k
        )
        
        # Filter and rank results
        filtered_results = [
            {
                "document": result[0],
                "score": result[1],
                "metadata": result[0].metadata
            }
            for result in retrieval_results
            if result[1] < 0.5  # Similarity threshold
        ]
        
        return filtered_results
    
    def generate_citations(self, retrieved_docs):
        citations = []
        for doc in retrieved_docs:
            metadata = doc['metadata']
            citation = self.format_mla_citation(metadata)
            citations.append({
                "text": doc['document'].page_content,
                "citation": citation
            })
        return citations
    
    def format_mla_citation(self, metadata):
        # Comprehensive MLA citation generation
        authors = metadata.get('authors', ['Unknown'])
        title = metadata.get('title', 'Untitled')
        publication = metadata.get('publication', 'Unknown Journal')
        year = metadata.get('year', 'n.d.')
        
        return f"{', '.join(authors)}. \"{title}.\" {publication}, {year}."

# Scientist
class Scientist:
    def __init__(self, knowledge_path):
        self.knowledge_base = ScientistKnowledgeBase(knowledge_path)
        self.knowledge_base.create_vectorstore()
        self.retriever = ScientistRetriever(self.knowledge_base)
        
        # Initialize LLM with strict temperature control
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3
        )
        
        # Update prompts to include confidence assessment
        self.analysis_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""Analyze the following research query using the provided context:
            Query: {query}
            Context: {context}
            
            Provide your analysis in the following format:
            ANALYSIS: [Your comprehensive analysis]
            CONFIDENCE: [Score between 0-100, based on:
            - Relevance of context to query (0-40 points)
            - Completeness of information (0-30 points)
            - Quality of sources (0-30 points)]
            REASONING: [Explain why you assigned this confidence score]
            """
        )
        
        self.fact_check_prompt = PromptTemplate(
            input_variables=["text", "context"],
            template="""Fact check the following text using the provided context:
            Text: {text}
            Context: {context}
            
            Provide your analysis in the following format:
            ANALYSIS: [Your detailed fact check]
            TRUE_CLAIMS: [List all verified true claims, or write "None" if no true claims found]
            FALSE_CLAIMS: [List any false or unverified claims, or write "None" if no false claims found]
            CORRECTIONS: [For each false claim, provide the correct information with citations. Format as: 
            Claim 1: [correct information]
            Claim 2: [correct information]
            etc.]
            CONFIDENCE: [Score between 0-100, based on:
            - Relevance of context (0-30 points)
            - Agreement with provided sources (0-40 points)
            - Accuracy of claims (0-30 points, deduct points proportionally to false claims)]
            REASONING: [Explain why you assigned this confidence score, mention both true and false claims]
            """
        )

    def _calculate_relevance_score(self, query, docs):
        """Calculate relevance score based on similarity scores."""
        embeddings = OpenAIEmbeddings()
        query_embedding = embeddings.embed_query(query)
        
        # Calculate cosine similarity for each document
        similarities = []
        for doc in docs:
            doc_embedding = embeddings.embed_query(doc.page_content)
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append(similarity)
        
        # Return normalized average similarity
        return float(np.mean(similarities))

    def _parse_llm_response(self, response):
        """Parse LLM response to extract analysis, confidence, reasoning, and claims."""
        parts = response.split('\n')
        analysis = ""
        confidence = 0.0
        reasoning = ""
        true_claims = []
        false_claims = []
        corrections = {}
        
        current_section = None
        correction_text = []
        
        for part in parts:
            if part.startswith("ANALYSIS:"):
                current_section = "ANALYSIS"
                analysis = part.replace("ANALYSIS:", "").strip()
            elif part.startswith("TRUE_CLAIMS:"):
                current_section = "TRUE_CLAIMS"
                claims = part.replace("TRUE_CLAIMS:", "").strip()
                if claims.lower() != "none":
                    true_claims = [claim.strip() for claim in claims.split(',')]
            elif part.startswith("FALSE_CLAIMS:"):
                current_section = "FALSE_CLAIMS"
                claims = part.replace("FALSE_CLAIMS:", "").strip()
                if claims.lower() != "none":
                    false_claims = [claim.strip() for claim in claims.split(',')]
            elif part.startswith("CORRECTIONS:"):
                current_section = "CORRECTIONS"
            elif part.startswith("CONFIDENCE:"):
                current_section = "CONFIDENCE"
                try:
                    confidence = float(part.replace("CONFIDENCE:", "").strip()) / 100.0
                    # Adjust confidence based on ratio of true to false claims
                    if false_claims:
                        total_claims = len(true_claims) + len(false_claims)
                        false_ratio = len(false_claims) / total_claims
                        confidence = confidence * (1 - (false_ratio * 0.7))  # Reduce confidence proportionally
                except ValueError:
                    confidence = 0.0
            elif part.startswith("REASONING:"):
                current_section = "REASONING"
                reasoning = part.replace("REASONING:", "").strip()
            elif current_section == "CORRECTIONS" and part.strip():
                correction_text.append(part.strip())
        
        # Process corrections
        for correction in correction_text:
            if correction.startswith("Claim"):
                parts = correction.split(":", 1)
                if len(parts) == 2:
                    claim_num = parts[0].strip()
                    correction_info = parts[1].strip()
                    corrections[claim_num] = correction_info
        
        return analysis, confidence, reasoning, true_claims, false_claims, corrections

    def analyze_query(self, query, top_k=5):
        """Analyze a research query with improved confidence scoring."""
        # Get relevant documents
        docs = self.knowledge_base.vectorstore.similarity_search(query, k=top_k)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(query, docs)
        
        # Create and run chain
        chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
        response = chain.run(query=query, context=context)
        
        # Parse response
        analysis, llm_confidence, reasoning = self._parse_llm_response(response)
        
        # Create detailed citations
        citations = []
        for i, doc in enumerate(docs):
            citation = {
                'citation': f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}",
                'content': doc.page_content[:200] + "...",  # Preview of content
                'relevance': f"{relevance_score:.2%}"
            }
            citations.append(citation)
        
        # Combine scores
        final_confidence = (0.7 * llm_confidence) + (0.3 * relevance_score)
        
        return {
            'analysis': analysis,
            'citations': citations,
            'source_count': len(docs),
            'confidence_score': final_confidence,
            'confidence_reasoning': reasoning
        }
    
    def fact_check(self, text):
        """Fact check with improved confidence scoring and claims analysis."""
        # Get relevant documents
        docs = self.knowledge_base.vectorstore.similarity_search(text, k=5)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Calculate relevance score
        relevance_score = self._calculate_relevance_score(text, docs)
        
        # Create and run chain
        chain = LLMChain(llm=self.llm, prompt=self.fact_check_prompt)
        response = chain.run(text=text, context=context)
        
        # Parse response with true claims
        analysis, llm_confidence, reasoning, true_claims, false_claims, corrections = self._parse_llm_response(response)
        
        # Create detailed citations
        citations = []
        for i, doc in enumerate(docs):
            citation = {
                'citation': f"Source {i+1}: {doc.metadata.get('source', 'Unknown')}",
                'content': doc.page_content[:200] + "...",  # Preview of content
                'relevance': f"{relevance_score:.2%}"
            }
            citations.append(citation)
        
        # Combine scores and adjust for false claims
        final_confidence = (0.7 * llm_confidence) + (0.3 * relevance_score)
        if false_claims:
            final_confidence = min(final_confidence, 0.3)  # Cap at 30% if false claims exist
        
        return {
            'fact_check_result': analysis,
            'citations': citations,
            'confidence_score': final_confidence,
            'confidence_reasoning': reasoning,
            'true_claims': true_claims,
            'false_claims': false_claims,
            'corrections': corrections
        }
