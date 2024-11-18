from transformers import pipeline

class TicketClassifier:
    def __init__(self, categories):
        self.classifier = pipeline(model="facebook/bart-large-mnli")
        self.categories = categories
    
    def categorize(self, desc):
        result = self.classifier(desc, candidate_labels=self.categories)
        print(result)
        if result['scores'][0] > 2 / len(self.categories):
            return result['labels'][0]
        else:
            return 'Other'