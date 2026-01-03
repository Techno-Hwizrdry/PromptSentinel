import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, Union

class PromptSentinel:
    def __init__(self, model_name: str) -> None:
        self.device = self.get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        ).to(self.device)
        self.model.eval()

    def get_device(self) -> torch.device:
        """
        Returns the best available device (CUDA for GPU or CPU).
        """
        return torch.device("cuda" if torch.cuda.is_available() else "cup")

    def scan(self, user_text: str,
             threat_threshold: float = 0.7) -> Dict[str, Union[str, float, bool]]:
        """
        Analyzes a string and returns a security report.
        """
        inputs = self.tokenizer(
            user_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Convert raw scores (logits) to probabilities
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Probability of class 1 (Injection)
            injection_score = probs[0][1].item()

        is_threat = injection_score > threat_threshold
        
        return {
            "text": user_text,
            "threat_score": round(injection_score, 4),
            "is_malicious": is_threat,
            "decision": "BLOCK" if is_threat else "PASS"
        }