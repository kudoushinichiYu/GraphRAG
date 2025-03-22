from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class HAPIRecommender:
    
    HAPI_columns = [
        'Humanitarian Need',
        'Refugee',
        'Returnee',
        'Operational Presence',
        'Funding',
        'Conflict Event',
        'National Risk',
        'Food Price',
        'Food Security',
        'Population',
        'Poverty Rate'
    ]

    
    def __init__(self, summary, output_number=None):
        self.summary = summary
        self.correlation_scores = []
        self.output_number = output_number
    
    def generate_recommendation(self):
        # Load pre-trained Sentence-BERT model
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and accurate

        def compute_sbert_similarity(word, article):
            embeddings = sbert_model.encode([word, article])  # Get embeddings
            return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]  # Compute similarity
        
        # Compute correlation scores
        for column in self.HAPI_columns:
            correlation_score = compute_sbert_similarity(column, self.summary)
            self.correlation_scores.append(correlation_score)
        
        self.correlation_scores = sorted(zip(self.HAPI_columns, self.correlation_scores), key=lambda x: x[1], reverse=True)
        recommendation_list = [col for col, _ in self.correlation_scores]
        
        if self.output_number:
            recommendation_list = recommendation_list[:self.output_number]
        return recommendation_list
        