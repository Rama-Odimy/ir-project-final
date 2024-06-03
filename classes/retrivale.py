import joblib
from scipy.sparse import csr_matrix
from scipy.sparse import csr_matrix
import numpy as np

class DocumentRetriever:
    def __init__(self, doc_file):
        # Load the fitted vectorizer
        self.vectorizer = joblib.load('D:/ir_final_final_final_the_flinalest/data/doc_vector_sparse.pkl')

        # Load the sparse matrix (assuming it's saved as a CSR matrix)
        # Convert to float64 during loading
        self.vsm = csr_matrix(joblib.load('D:/ir_final_final_final_the_flinalest/data/doc_vector_sparse.pkl').dtype(np.float64)) 

        # Load the mapping between document IDs and line numbers
        self.doc_id_to_line_number = self.load_doc_id_mapping(doc_file)

    def load_doc_id_mapping(self, doc_file):
        doc_id_to_line_number = {}
        with open(doc_file, 'r') as f:
            for line_number, line in enumerate(f):
                parts = line.strip().split('\t')  # Assuming tab as delimiter
                if len(parts) == 2:
                    document_id, text = parts
                    doc_id_to_line_number[document_id] = line_number
                else:
                    print(f"Warning: Line {line_number + 1} does not have a tab delimiter: {line.strip()}")
        return doc_id_to_line_number

    def get_doc_id_from_row_index(self, row_index):
        """
        Retrieves the document ID corresponding to a given row index.

        Args:
            row_index: The row index in the sparse matrix.

        Returns:
            The document ID, or None if the row index is not found in the mapping.
        """

        for doc_id, index in self.doc_id_to_line_number.items():
            if index == row_index:
                return doc_id
        return None

# Example usage:
doc_file = 'D:/ir_final_final_final_the_flinalest/data/doc_id_mapping.txt' # Replace with your actual file path
retriever = DocumentRetriever(doc_file)

# Get a row index (e.g., from your cosine similarity calculations)
row_index = 5  # Example row index

# Retrieve the document ID
doc_id = retriever.get_doc_id_from_row_index(row_index)

# Print the document ID
print(f"Document ID for row index {row_index}: {doc_id}")