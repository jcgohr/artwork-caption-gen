from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AutoFewShot:
    def __init__(self, construct_data:list[tuple]):
        """
        Args:
            construct_data: List[tuple] of format [(text, 1|0)], 1 for visual, 0 for contextual
        """
        self.vectorizer = TfidfVectorizer()
        self.data = construct_data
        self.database = self._construct_database(construct_data)

    def _construct_database(self, construct_data):
        """
        Construct a new text database
        """
        
        tfidf_matrix = self.vectorizer.fit_transform([data[0] for data in construct_data])
        return tfidf_matrix
        
    def most_similar(self, text:str, top_n:int=1)->list[tuple]:
        """
        Return the most similar texts from the database of example texts

        Args:
            text: Input text to search with
            top_n: Number of texts to return
        """
        query_tfidf = self.vectorizer.transform([text])
        similarities = cosine_similarity(query_tfidf, self.database)[0]
        ranked_indices = similarities.argsort()[::-1][:top_n]
        return [self.data[idx] for idx in ranked_indices]