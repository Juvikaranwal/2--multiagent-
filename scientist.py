import streamlit as st
from scientistRAG import Scientist
from dotenv import load_dotenv
import os

# Move initialization outside the class
@st.cache_resource
def initialize_scientist():
    """Initialize the Scientist agent"""
    load_dotenv()
    knowledge_path = os.getenv('KNOWLEDGE_BASE_PATH')
    return Scientist(knowledge_path)

class ScientistApp:
    def __init__(self):
        self.setup_page_config()
        self.setup_styles()
        self.scientist = initialize_scientist()

    def setup_page_config(self):
        st.set_page_config(
            page_title="Scientist",
            page_icon="🧬",
            layout="wide"
        )

    def setup_styles(self):
        st.markdown("""
            <style>
            .main {
                padding: 2rem;
            }
            .stAlert {
                padding: 1rem;
                margin: 1rem 0;
            }
            .citation {
                font-size: 0.9em;
                color: #666;
                padding: 0.5rem;
                border-left: 3px solid #666;
                margin: 0.5rem 0;
            }
            .source-box {
                background-color: #f0f2f6;
                padding: 1rem;
                border-radius: 0.5rem;
                margin: 0.5rem 0;
            }
            </style>
        """, unsafe_allow_html=True)

    def run(self):
        try:
            # Header
            st.title("Scientist")
            st.markdown("---")
            
            # Main content - Fact Checking
            st.subheader("✔️ Fact and Accuracy Checking")
            
            # Add file upload option
            check_method = st.radio(
                "Choose input method:",
                ["Enter Text", "Upload Document"]
            )
            
            if check_method == "Upload Document":
                uploaded_file = st.file_uploader(
                    "Upload a document to fact-check",
                    type=['txt', 'pdf', 'docx']
                )
                
                if uploaded_file is not None:
                    try:
                        if uploaded_file.type == "application/pdf":
                            text_to_check = self.read_pdf(uploaded_file)
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text_to_check = self.read_docx(uploaded_file)
                        else:  # txt files
                            text_to_check = uploaded_file.getvalue().decode()
                        
                        st.text_area(
                            "Extracted text:",
                            text_to_check,
                            height=150,
                            disabled=True
                        )
                        
                        if st.button("Fact Check Document"):
                            with st.spinner("Fact checking..."):
                                self.process_fact_check(text_to_check)
                    except Exception as e:
                        st.error(f"Error processing file: {str(e)}")
            
            else:  # Enter Text option
                text_to_check = st.text_area(
                    "Enter text to fact-check:",
                    height=150,
                    placeholder="Enter the text you want to verify..."
                )
                
                if st.button("Fact Check"):
                    if text_to_check:
                        with st.spinner("Fact checking..."):
                            self.process_fact_check(text_to_check)
                    else:
                        st.warning("Please enter text to fact-check.")
            
            # Footer
            st.markdown("---")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please check your environment configuration and try again.")

    def read_pdf(self, file):
        """Extract text from PDF file."""
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

    def read_docx(self, file):
        """Extract text from DOCX file."""
        import docx2txt
        text = docx2txt.process(file)
        return text

    def process_fact_check(self, text):
        try:
            result = self.scientist.fact_check(text)
            
            # Display results with color based on accuracy AND relevance
            st.markdown("### Fact Check Results")
            
            # Check if the content is irrelevant or contains false claims
            is_irrelevant = "not contain any information" in result['fact_check_result'].lower() or \
                           "irrelevant" in result['fact_check_result'].lower()
            
            if is_irrelevant or result.get('false_claims'):
                st.error(result['fact_check_result'])    # Red box for irrelevant or inaccurate results
            else:
                st.success(result['fact_check_result'])  # Green box only for relevant AND accurate results
            
            # If there are false claims, show corrections in a single block
            if result.get('false_claims'):
                st.markdown("### Corrected Information")
                corrections = list(result.get('corrections', {}).values())
                combined_corrections = " ".join(corrections)
                st.info(combined_corrections)
            
            # Display confidence score
            st.markdown("---")
            confidence_score = result['confidence_score']
            st.metric("Confidence Score", f"{confidence_score:.2%}")
            
            # Display supporting citations
            st.markdown("### Supporting Sources")
            for citation in result['citations']:
                with st.expander(f"Source: {citation['citation']}"):
                    st.write("**Content Preview:**")
                    st.write(citation['content'])
                    st.write(f"**Relevance Score:** {citation['relevance']}")
                    
        except Exception as e:
            st.error(f"Error during fact checking: {str(e)}")

def main():
    app = ScientistApp()
    app.run()

if __name__ == "__main__":
    main() 
