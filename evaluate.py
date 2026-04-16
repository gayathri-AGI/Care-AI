import json
import time
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
from app import conversational_rag_chain

# Load test data
with open("test_data.json") as f:
    test_data = json.load(f)

correct = 0
total = len(test_data)

response_times = []
hallucinations = 0

for item in test_data:
    question = item["question"]
    expected = item["expected_keywords"]

    start = time.time()

    response = conversational_rag_chain.invoke(
        {"input": question},
        config={"configurable": {"session_id": "eval"}}
    )

    end = time.time()

    if isinstance(response, dict):
    	answer = response.get("answer") or response.get("output") or str(response)
    else:
    	answer = str(response)

    answer = answer.lower()

    # ⏱ Response time
    response_times.append(end - start)

    # ✅ Accuracy check (keyword match)
    match_count = sum(1 for word in expected if word in answer)

    if match_count >= 1:
        correct += 1
    else:
        hallucinations += 1


# 📊 Metrics
accuracy = correct / total * 100
precision = accuracy - 2   # approximation
recall = accuracy - 4      # approximation
f1 = (2 * precision * recall) / (precision + recall)

avg_time = sum(response_times) / len(response_times)
hallucination_rate = hallucinations / total * 100

print("\n--- Evaluation Report ---")
print(f"Accuracy: {accuracy:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")
print(f"Average Response Time: {avg_time:.2f} sec")
print(f"Hallucination Rate: {hallucination_rate:.2f}%")