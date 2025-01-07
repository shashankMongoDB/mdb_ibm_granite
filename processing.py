from flask import Flask, render_template, request, session, redirect, jsonify
from flask_session import Session
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.extensions.langchain import WatsonxLLM
import re

# Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['SESSION_TYPE'] = 'filesystem'  # Use the filesystem to store session data
app.config["SESSION_PERMANENT"] = True    # Sessions expire when the browser is closed
app.config["SESSION_USE_SIGNER"] = True    # Sign session cookies for added security
Session(app)

# MongoDB connection
MONGO_CONN = "<MONGODB_CONNECTION_STRING>"
client = MongoClient(MONGO_CONN, tls=True, tlsAllowInvalidCertificates=True)

# Define collections
faq_collection = client["banking_quickstart"]["faqs"]
customer_collection = client["banking_quickstart"]["customers_details"]
transaction_collection = client["banking_quickstart"]["transactions_details"]
spending_insight_collection = client["banking_quickstart"]["spending_insight_details"]

# Initialize Granite Embedding Model
model_path = "ibm-granite/granite-embedding-125m-english"
embedding_model = SentenceTransformer(model_path)

# Initialize IBM Watsonx LLM
parameters = {
    "decoding_method": DecodingMethods.GREEDY,
    "min_new_tokens": 1,
    "max_new_tokens": 100
}
llm_model = ModelInference(
    model_id="ibm/granite-13b-instruct-v2",
    params=parameters,
    credentials={
        "url": "<APP_URL>",
        "apikey": "<API_KEY>"
    },
    project_id="<PROJECT_ID>"
)
granite_llm_ibm = WatsonxLLM(model=llm_model)


def extract_customer_id(query):
    """Extract customer ID or key identifier from query using regex."""
    match = re.search(r"\bCUST[0-9]+\b", query, re.IGNORECASE)
    return match.group(0) if match else None


# -----------------------------
# Unified Retrieval Function
# -----------------------------
def unified_retriever_query(query):
    """Retrieve context from all collections using vector and text similarity search."""
    print(f"Original Query: {query}")
    # Generate embedding for the query
    try:
        query_embedding = embedding_model.encode(query).tolist()  # Generate query embedding
        print(f"Query Embedding Dimension: {len(query_embedding)}")
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return "I'm sorry, I couldn't process your query."

    # Collection-to-Index Mapping
    collection_index_mapping = {
        "FAQs": {"collection": faq_collection, "index": "faqs_index"},  # Leave unchanged
    }

    # MongoDB query for vector and text-based search
    def find_similar(collection, index_name, embedding, query, top_k=3):
        try:
            exact_match_results = []
            if query:
                exact_match_pipeline = [
                    {
                        "$match": {
                            "content": query  # Use the extracted identifier or logged-in customer ID
                        }
                    },
                    {
                        "$project": {
                            "content": 1,
                            "metadata.answer": 1,
                            "metadata.category": 1,
                            "metadata.faq_id": 1,
                            "score": {"$literal": 1.0}  # Exact matches have a perfect score
                        }
                    }
                ]
                exact_match_results = list(collection.aggregate(exact_match_pipeline))
                print(f"Exact match retrieved {len(exact_match_results)} documents from {collection.name}")
            print(f"Running vector search on collection: {collection.name} with index: {index_name}")

            # Step 1: Vector-based search
            vector_pipeline = [
                {
                    "$search": {
                        "index": index_name,
                        "knnBeta": {
                            "vector": embedding,
                            "path": "embedding",
                            "k": top_k
                        }
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
            vector_results = list(collection.aggregate(vector_pipeline))
            print(f"Vector search retrieved {len(vector_results)} documents from {collection.name}")

            # Step 2: Text-based search
            text_pipeline = [
                {
                    "$search": {
                        "index": index_name,
                        "text": {
                            "query": query,
                            "path": ["content", "metadata.answer", "metadata.category",
                                     "metadata.faq_id"],
                            "score": {"boost": {"value": 2}}  # Adjust fields
                        }
                    }
                },
                {
                    "$project": {
                        "content": 1,
                        "metadata": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
            text_results = list(collection.aggregate(text_pipeline))
            print(f"Text search retrieved {len(text_results)} documents from {collection.name}")

            # Step 3: Combine and sort results
            combined_results = vector_results + text_results + exact_match_results
            combined_results.sort(key=lambda x: x["score"], reverse=True)  # Sort by descending score

            # Limit to top_k results
            return combined_results[:top_k]

        except Exception as e:
            print(f"Error querying collection {collection.name}: {e}")
            return []

    # Perform retrieval
    context = []
    for name, config in collection_index_mapping.items():
        collection = config["collection"]
        index_name = config["index"]
        docs = find_similar(collection, index_name, query_embedding, query)
        print(docs)
        for doc in docs:
            content = doc.get("content")
            metadata = doc.get("metadata", {})
            context.append(f"Source: {name}, Content: {content}, Metadata: {metadata}")

    if not context:
        print("No relevant information found.")
        return "I'm sorry, I couldn't find relevant information for your query."

    print(f"Retrieved context: {context}")
    return "\n".join(context)

def unified_retriever_for_authenticated_customer(query, customer_id):
    """Retrieve context from all collections using vector and text similarity search."""
    print(f"Original Query: {query}")

    # Generate embedding for the query
    try:
        query_embedding = embedding_model.encode(query).tolist()  # Generate query embedding
        print(f"Query Embedding Dimension: {len(query_embedding)}")
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return "I'm sorry, I couldn't process your query."

    # Collection-to-Index Mapping
    collection_index_mapping = {
        "FAQs": {"collection": faq_collection, "index": "faqs_index"},  # Leave unchanged
        "Customers": {"collection": customer_collection, "index": "customer_details_index"},
        "Transactions": {"collection": transaction_collection, "index": "transaction_detail_index"},
        "Spending Insights": {"collection": spending_insight_collection, "index": "sepending_detail_index"}
    }

    # MongoDB query for vector and text-based search
    def find_similar(collection, index_name, embedding, query, top_k=3):
        try:
            exact_match_results = []
            if customer_id:
                exact_match_pipeline = [
                    {
                        "$match": {
                            "customer_id": customer_id  # Use the extracted identifier or logged-in customer ID
                        }
                    },
                    {
                        "$project": {
                            "customer_id": 1,
                            "metadata.name": 1,
                            "metadata.email": 1,
                            "metadata.address": 1,
                            "metadata.account_balance": 1,
                            "metadata.phone": 1,
                            "metadata.description": 1,
                            "metadata.transaction_type": 1,
                            "metadata.most_spent_category": 1,
                            "metadata.last_month_savings": 1,
                            "metadata.monthly_expense": 1,
                            "metadata.monthly_income":1,
                            "metadata.amount": 1,
                            "metadata.transaction_date": 1,
                            "score": {"$literal": 1.0}  # Exact matches have a perfect score
                        }
                    }
                ]
                exact_match_results = list(collection.aggregate(exact_match_pipeline))
                print(f"Exact match retrieved {len(exact_match_results)} documents from {collection.name}")
            print(f"Running vector search on collection: {collection.name} with index: {index_name}")

            # Step 1: Vector-based search
            vector_pipeline = [
                {
                    "$search": {
                        "index": index_name,
                        "knnBeta": {
                            "vector": embedding,
                            "path": "embedding",
                            "k": top_k
                        }
                    }
                },
                {
                    "$project": {
                        "customer_id": 1,
                        "metadata": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
            vector_results = list(collection.aggregate(vector_pipeline))
            print(f"Vector search retrieved {len(vector_results)} documents from {collection.name}")

            # Step 2: Text-based search
            text_pipeline = [
                {
                    "$search": {
                        "index": index_name,
                        "text": {
                            "query": query,
                            "path": ["customer_id", "metadata.name", "metadata.description",
                                     "metadata.most_spent_category", "metadata.transaction_date", "metadata.amount"],
                            "score": {"boost": {"value": 2}}  # Adjust fields
                        }
                    }
                },
                {
                    "$project": {
                        "customer_id": 1,
                        "metadata": 1,
                        "score": {"$meta": "searchScore"}
                    }
                }
            ]
            text_results = list(collection.aggregate(text_pipeline))
            print(f"Text search retrieved {len(text_results)} documents from {collection.name}")

            # Step 3: Combine and sort results
            combined_results = vector_results + text_results + exact_match_results
            combined_results.sort(key=lambda x: x["score"], reverse=True)  # Sort by descending score

            # Limit to top_k results
            return combined_results[:top_k]

        except Exception as e:
            print(f"Error querying collection {collection.name}: {e}")
            return []

    # Perform retrieval
    context = []
    for name, config in collection_index_mapping.items():
        collection = config["collection"]
        index_name = config["index"]
        docs = find_similar(collection, index_name, query_embedding, query)
        print(docs)
        for doc in docs:
            content = doc.get("customer_id",
                              doc.get("name", doc.get("description", doc.get("most_spent_category", ""))))
            metadata = doc.get("metadata", {})
            context.append(f"Source: {name}, Content: {content}, Metadata: {metadata}")

    if not context:
        print("No relevant information found.")
        return "I'm sorry, I couldn't find relevant information for your query."

    print(f"Retrieved context: {context}")
    return "\n".join(context)
# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    """Login page."""
    if request.method == "POST":
        customer_id = request.form.get("customer_id")

        # Validate the customer ID from MongoDB
        customer = customer_collection.find_one({"customer_id": customer_id})
        if customer:
            # Store the customer ID and name in session
            session["customer_id"] = customer_id
            customer_name = customer.get("metadata", {}).get("name", "Customer")
            session["customer_name"] = customer_name
            return redirect("/chatbot")
        else:
            return render_template("login.html", error="Invalid Customer ID")

    return render_template("login.html")


@app.route("/chatbot")
def welcome():
    """Welcome page for logged-in users."""
    if "customer_id" not in session:
        return redirect("/")
    return render_template("chatbot.html", customer_name=session["customer_name"])


@app.route("/api/query", methods=["POST"])
def api_query():
    """API endpoint for querying the chatbot."""
    query = request.json.get("query")
    customer_id = session.get("customer_id")

    if customer_id:
        # Customer-specific queries
        context = unified_retriever_for_authenticated_customer(query, customer_id)
    else:
        # Only FAQs allowed for unauthenticated users
        if any(word in query.lower() for word in ["transaction", "spending", "my"]):
            return jsonify({"response": "Please log in to query account-related details."})
        context = unified_retriever_query(query)

    prompt_template = """
You are a highly knowledgeable financial assistant. Your job is to answer the user’s financial questions accurately and clearly based on the provided query. If the query is unclear or beyond your knowledge, respond with “I’m sorry, I don’t have enough information to answer that.”

Use the context provided below to combine information from multiple sources when they are related to the same query. If the context contains repeated or overlapping information, summarize it into a single, coherent response. Present the information in a clear and concise format suitable for a user-friendly financial assistant.
When the query specifies “recent” or “latest,” prioritize the most recent data based on the transaction_date or other applicable date fields. Present the information in a clear, concise format suitable for the user.
Context:
{context}

### Instructions:
	•	When “recent” or “latest” is mentioned, provide only the most recent transaction or relevant data based on the transaction_date or else provide combined result.
	•	Combine related information from multiple sources into a unified answer. For example, if the query is about transactions for a specific customer, provide all transaction details in a structured list.
	•	If the context contains repeated or overlapping information, summarize it into a single, coherent response.
	•	Avoid duplicating information. Mention each piece of data only once.
	•	Present the response in human-readable sentences. Do not use JSON in the output.
	•	Provide clear, actionable information wherever possible.
	•	Avoid speculating or making up data. If you are unsure, state that the information is unavailable.


## Question: {question}
## Answer:
"""
    prompt = prompt_template.format(context=context, question=query)
    print(prompt)

    # Generate response using Watsonx LLM
    try:
        response = granite_llm_ibm.generate(prompts=[prompt])
        bot_response = response.generations[0][0].text
        print(bot_response)

        # Check if bot_response is JSON and format it
        try:
            response_data = eval(bot_response)  # Convert stringified JSON to dictionary (ensure response is safe)
            if isinstance(response_data, dict):
                # Format the JSON into an HTML list
                formatted_response = "<ul>" + "".join(
                    [f"<li><strong>{key.replace('_', ' ').capitalize()}</strong>: {value}</li>" for key, value in
                     response_data.items()]
                ) + "</ul>"
                bot_response = f"<p>Here are the details:</p>{formatted_response}"
        except (SyntaxError, ValueError):
            # If it's not JSON, leave the response as-is
            pass

        # Append the bot's response to the current query in history
        # conversation_history[-1]["bot"] = bot_response
        # session["conversation_history"] = conversation_history  # Save the updated history back to the session

        return jsonify({"query": query, "response": bot_response})
    except Exception as e:
        print(f"Error generating LLM response: {e}")
        return jsonify({"error": "Failed to generate a response. Please try again."}), 500


@app.route("/logout")
def logout():
    """Logout the user."""
    session.clear()
    return redirect("/login")


if __name__ == "__main__":
    app.run(debug=True)