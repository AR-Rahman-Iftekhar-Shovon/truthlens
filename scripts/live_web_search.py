import os
import json
import urllib.request

def search_web(query):
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not found")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "max_results": 5,
        "include_answer": True
    }

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    print("\nSEARCH ANSWER:")
    print(data.get("answer") or "No summary available")

    print("\nSOURCES:")
    for i, result in enumerate(data.get("results", []), 1):
        print(f"\n{i}. {result.get('title')}")
        print("URL:", result.get("url"))
        print("CONTENT:", result.get("content", "")[:500])

if __name__ == "__main__":
    query = input("Enter a claim to research: ")
    search_web(query)
