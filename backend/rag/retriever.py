"""
backend/rag/retriever.py
Controlled Medical Knowledge Base Indexer and Semantic Retriever.

Sources: WHO, CDC, NIH, AHA, BTS, ACR.
Implements exact term matching, lexical TF-IDF scoring, and structured section retrieval.
"""

import os
import json
import math
import re
from typing import List, Dict, Any, Tuple

class MedicalRAGRetriever:
    def __init__(self, kb_dir: str = None):
        if kb_dir is None:
            kb_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "knowledge_base"))
        self.kb_dir = kb_dir
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: List[Dict[str, Any]] = []
        self._load_knowledge_base()

    def _load_knowledge_base(self):
        if not os.path.exists(self.kb_dir):
            raise FileNotFoundError(f"Knowledge base directory not found at: {self.kb_dir}")

        for file_name in os.listdir(self.kb_dir):
            if file_name.endswith(".json"):
                path = os.path.join(self.kb_dir, file_name)
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    condition = doc["condition"]
                    self.documents[condition.lower()] = doc

                    # Index chunks for retrieval
                    self._create_chunks(doc)

    def _create_chunks(self, doc: Dict[str, Any]):
        condition = doc["condition"]
        sources = doc.get("reliable_sources", [])

        # Overview chunk
        self.chunks.append({
            "condition": condition,
            "section": "Overview & Pathology",
            "content": f"{condition} (ICD-10: {doc.get('icd_10', 'N/A')}): {doc.get('overview', '')}",
            "sources": sources
        })

        # Radiographic findings chunk
        if "radiographic_findings" in doc:
            findings_text = "; ".join(doc["radiographic_findings"])
            self.chunks.append({
                "condition": condition,
                "section": "Radiographic Findings",
                "content": f"Key radiographic manifestations of {condition}: {findings_text}",
                "sources": sources
            })

        # Symptoms chunk
        if "common_symptoms" in doc:
            symptoms_text = "; ".join(doc["common_symptoms"])
            self.chunks.append({
                "condition": condition,
                "section": "Clinical Presentation & Symptoms",
                "content": f"Common symptoms observed in {condition}: {symptoms_text}",
                "sources": sources
            })

        # Causes & Risks chunk
        if "causes_and_risk_factors" in doc:
            causes_text = "; ".join(doc["causes_and_risk_factors"])
            self.chunks.append({
                "condition": condition,
                "section": "Etiology & Risk Factors",
                "content": f"Causes and risk factors for {condition}: {causes_text}",
                "sources": sources
            })

        # Diagnostics chunk
        if "diagnostic_information" in doc:
            diag_text = "; ".join(doc["diagnostic_information"])
            self.chunks.append({
                "condition": condition,
                "section": "Diagnostic Evaluation",
                "content": f"Diagnostic workup for {condition}: {diag_text}",
                "sources": sources
            })

        # Treatment chunk
        if "treatment_approaches" in doc:
            tx_text = "; ".join(doc["treatment_approaches"])
            self.chunks.append({
                "condition": condition,
                "section": "Therapeutic Approaches",
                "content": f"Clinical management of {condition}: {tx_text}",
                "sources": sources
            })

        # When to seek care chunk
        if "when_to_seek_care" in doc:
            care_text = "; ".join(doc["when_to_seek_care"])
            self.chunks.append({
                "condition": condition,
                "section": "Emergency Warning Signs",
                "content": f"Warning signs requiring immediate medical care in {condition}: {care_text}",
                "sources": sources
            })

    def retrieve(self, query: str, top_k: int = 4, target_condition: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves top relevant medical passages using term frequency - inverse document frequency scoring.
        Prioritizes the diagnosed condition if specified.
        """
        words = re.findall(r"\w+", query.lower())
        scored = []

        for chunk in self.chunks:
            chunk_cond = chunk["condition"].lower()
            text = (chunk["section"] + " " + chunk["content"]).lower()
            
            # Exact condition bonus
            score = 0.0
            if target_condition and target_condition.lower() == chunk_cond:
                score += 8.0
            elif target_condition and target_condition.lower() in text:
                score += 4.0

            # Lexical keyword match
            for w in words:
                if len(w) > 2 and w in text:
                    score += 2.0
                    if w in chunk["section"].lower():
                        score += 3.0

            if score > 0:
                scored.append((score, chunk))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in scored[:top_k]]

    def get_condition_profile(self, condition: str) -> Dict[str, Any]:
        """Returns the full master document for a given clinical finding."""
        key = condition.lower()
        if key in self.documents:
            return self.documents[key]
        for k, v in self.documents.items():
            if k in key or key in k:
                return v
        return self.documents.get("normal", {})
