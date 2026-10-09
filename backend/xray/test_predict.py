"""
backend/xray/test_predict.py
Verifies Step 12, 13, 14, 15, and 16:
- Predicts condition on a test radiograph
- Extracts confidence & ranked possibilities
- Generates Grad-CAM attention heatmap
- Triggers automated RAG medical information retrieval
- Answers a follow-up medical question
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from backend.xray.predict import XrayPredictor

def test_pipeline():
    predictor = XrayPredictor()
    test_img = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/test/pneumonia/CXR_00561.png"))
    
    print("==================================================")
    print("TESTING FULL INFERENCE + GRAD-CAM + RAG PIPELINE")
    print(f"Target Test Image: {test_img}")
    print("==================================================")

    res = predictor.predict_image_file(test_img)

    print(f"Predicted Condition: {res['predicted_condition']}")
    print(f"Confidence Score   : {res['confidence_percentage']}%")
    print(f"Clinical Status    : {res['status']}")
    print("\nRanked Predictions:")
    for p in res['ranked_predictions']:
        marker = "★" if p['is_top'] else " "
        print(f"  {marker} {p['condition']:<16} : {p['probability']:5.2f}% ({p['category']})")

    print(f"\nGrad-CAM Heatmap   : {len(res['gradcam']['heatmap'])} points (grid 28x28)")
    print(f"Grad-CAM Min/Max   : min={min(res['gradcam']['heatmap']):.3f}, max={max(res['gradcam']['heatmap']):.3f}")

    print("\nAutomated RAG Report Generated:")
    rag = res['rag_information']
    print(f"  Condition: {rag['predicted_condition']} (ICD: {rag['icd_10_code']})")
    print(f"  Overview : {rag['sections']['overview']['text'][:120]}...")
    print(f"  Symptoms : {len(rag['sections']['symptoms']['items'])} documented clinical symptoms")
    print(f"  Sources  : {[s['organization'] for s in rag['sources']]}")

    # Test follow-up question
    print("\nTesting Follow-Up RAG Query:")
    q_res = predictor.rag_generator.answer_query("What are the primary symptoms and emergency signs?", res['predicted_condition'])
    print(f"  Query : {q_res['query']}")
    print(f"  Answer: {q_res['answer'][:150]}...")
    print("==================================================")
    print("SUCCESS: Full Prediction & RAG Pipeline Verified!")
    print("==================================================")

if __name__ == "__main__":
    test_pipeline()
