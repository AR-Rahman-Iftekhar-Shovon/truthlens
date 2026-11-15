import json
import os
import socket
import time
import urllib.error
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
        "include_answer": True,
    }

    req = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    # Retry up to 3 times
    for attempt in range(1, 4):
        try:
            print("\nSearching.....")

            with urllib.request.urlopen(req, timeout=45) as response:
                data = json.loads(response.read().decode("utf-8"))

            print("\nSEARCH ANSWER:")
            print(data.get("answer") or "No summary available")

            print("\nSOURCES:")

            for i, result in enumerate(data.get("results", []), 1):
                print(f"\n{i}. {result.get('title')}")
                print("URL:", result.get("url"))
                print("CONTENT:", result.get("content", "")[:500])

            return data

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            print(f"\nHTTP Error: {e.code}")
            print("Details:", error_body[:1000])

            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(2**attempt)
                continue

            return None

        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            print(f"\nConnection error: {e}")

            if attempt < 3:
                wait = 2**attempt
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                print("\nsearch failed after 3 attempts.")
                return None

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"\nInvalid API response: {e}")
            return None


if __name__ == "__main__":
    query = input("Enter a claim to research: ").strip()

    if query:
        search_web(query)
    else:
        print("Please enter a claim.")
