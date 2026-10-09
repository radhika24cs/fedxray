"""
backend/rag/generator.py
RAG Generation Engine for Medical Information Retrieval.

Adheres to rigorous clinical communication standards:
- Always states 'The model predicts...' rather than 'You have...'
- Synthesizes retrieved evidence directly from controlled WHO, CDC, and NIH knowledge sources
- Strictly provides non-diagnostic educational information with clear reference attribution
"""

from typing import Dict, Any, List
from backend.rag.retriever import MedicalRAGRetriever

DISCLAIMER = (
    "DISCLAIMER: This system is an AI research and educational demonstration "
    "developed for Privacy-Preserving Federated Learning research. It does NOT "
    "constitute medical advice, diagnosis, or clinical management. Consult a licensed "
    "physician or radiologist for all health decisions."
)

class MedicalRAGGenerator:
    def __init__(self, retriever: MedicalRAGRetriever = None):
        if retriever is None:
            retriever = MedicalRAGRetriever()
        self.retriever = retriever

    def generate_condition_report(self, predicted_condition: str, confidence: float) -> Dict[str, Any]:
        """
        Synthesizes the comprehensive 6-section medical overview from retrieved documents.
        """
        doc = self.retriever.get_condition_profile(predicted_condition)
        condition_name = doc.get("condition", predicted_condition)
        
        return {
            "predicted_condition": condition_name,
            "confidence_percentage": round(confidence * 100, 2),
            "icd_10_code": doc.get("icd_10", "Unspecified"),
            "category": doc.get("category", "General Cardiopulmonary"),
            "clinical_disclaimer": DISCLAIMER,
            "sections": {
                "overview": {
                    "title": "1. What the Predicted Condition Means",
                    "text": f"The federated model prediction indicates patterns consistent with {condition_name}. {doc.get('overview', '')}"
                },
                "symptoms": {
                    "title": "2. Common Symptoms",
                    "items": doc.get("common_symptoms", [])
                },
                "causes_and_risks": {
                    "title": "3. Common Causes & Risk Factors",
                    "items": doc.get("causes_and_risk_factors", [])
                },
                "diagnostics": {
                    "title": "4. Diagnostic Evaluation",
                    "items": doc.get("diagnostic_information", []),
                    "radiographic_findings": doc.get("radiographic_findings", [])
                },
                "treatment": {
                    "title": "5. General Treatment Approaches",
                    "items": doc.get("treatment_approaches", [])
                },
                "when_to_seek_care": {
                    "title": "6. When to Seek Medical Attention",
                    "items": doc.get("when_to_seek_care", [])
                }
            },
            "sources": doc.get("reliable_sources", [])
        }

    def answer_query(self, user_query: str, predicted_condition: str) -> Dict[str, Any]:
        """
        Answers interactive follow-up questions using retrieved context chunks.
        """
        retrieved_chunks = self.retriever.retrieve(
            query=user_query,
            top_k=3,
            target_condition=predicted_condition
        )

        doc = self.retriever.get_condition_profile(predicted_condition)
        condition_name = doc.get("condition", predicted_condition)

        # Build grounded synthesis
        excerpts = []
        sources = []
        for ch in retrieved_chunks:
            excerpts.append(f"[{ch['section']}]: {ch['content']}")
            for src in ch.get("sources", []):
                if src not in sources:
                    sources.append(src)

        # Context-aware response template
        query_lower = user_query.lower()
        if any(w in query_lower for w in ["symptom", "feel", "sign", "presentation"]):
            body = (
                f"Based on clinical documentation regarding {condition_name}, commonly reported symptoms include: "
                + "; ".join(doc.get("common_symptoms", [])) + "."
            )
        elif any(w in query_lower for w in ["cause", "risk", "why", "factor", "etiology"]):
            body = (
                f"Documented causes and predisposing risk factors for {condition_name} include: "
                + "; ".join(doc.get("causes_and_risk_factors", [])) + "."
            )
        elif any(w in query_lower for w in ["treat", "cure", "medicine", "drug", "care", "therapy"]):
            body = (
                f"General clinical management approaches for {condition_name} typically involve: "
                + "; ".join(doc.get("treatment_approaches", [])) + "."
            )
        elif any(w in query_lower for w in ["emergency", "hospital", "urgent", "doctor", "seek", "danger"]):
            body = (
                f"Medical attention is urgently indicated if the patient experiences: "
                + "; ".join(doc.get("when_to_seek_care", [])) + "."
            )
        else:
            body = (
                f"Regarding your query about {condition_name}, medical guidelines state that: "
                + doc.get("overview", "") + " "
                + "Consult with your healthcare provider for clinical evaluation."
            )

        return {
            "query": user_query,
            "predicted_condition": condition_name,
            "answer": body,
            "grounded_excerpts": excerpts,
            "citations": sources,
            "disclaimer": DISCLAIMER
        }
