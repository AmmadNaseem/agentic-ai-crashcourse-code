import base64
import io
import json
import os
import sqlite3
from typing import Optional

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from PIL import Image, ImageOps
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from reviews_api import get_product_rating

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "store.db")

# Separate overrides let text and image models be changed independently.
MODEL_NAME = os.getenv("SHOPPING_MODEL", "qwen/qwen3.8-27b")
VISION_MODEL_NAME = os.getenv("SHOPPING_VISION_MODEL", "qwen/qwen3.8-27b")
OUTPUT_TOKEN_LIMIT = int(os.getenv("SHOPPING_OUTPUT_TOKEN_LIMIT", "1000"))
MAX_TOKENS = min(int(os.getenv("SHOPPING_MAX_TOKENS", "512")), OUTPUT_TOKEN_LIMIT)
if MAX_TOKENS < 1:
    raise ValueError("Shopping output token limits must be positive.")

llm = ChatGroq(
    model=MODEL_NAME,
    temperature=0,
    max_tokens=MAX_TOKENS,
    reasoning_effort="none",
    reasoning_format="hidden",
)
vision_llm = ChatGroq(
    model=VISION_MODEL_NAME,
    temperature=0,
    max_tokens=MAX_TOKENS,
    reasoning_effort="none",
    reasoning_format="hidden",
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def search_products(query: str, max_price: Optional[float] = None, is_organic: Optional[bool] = None) -> str:
    """
    Search the product database by keyword (matched against name, description, and category).
    Optionally filter by maximum price and/or organic status.
    Returns a JSON array of matching products, each with: id, name, category, price,
    description, is_organic.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    sql = "SELECT id, name, category, price, description, is_organic FROM products WHERE 1=1"
    #why use 1=1? 
    # It allows us to append conditions dynamically without worrying about whether it's the first condition or not.
    params: list = []

    if query:
        sql += " AND (name LIKE ? OR description LIKE ? OR category LIKE ?)"
        like = f"%{query}%"
        params.extend([like, like, like])

    if max_price is not None:
        sql += " AND price <= ?"
        params.append(max_price)

    if is_organic is not None:
        sql += " AND is_organic = ?"
        params.append(1 if is_organic else 0)

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    products = [
        {
            "id":          row[0],
            "name":        row[1],
            "category":    row[2],
            "price":       row[3],
            "description": row[4],
            "is_organic":  bool(row[5]),
        }
        for row in rows
    ]
    return json.dumps(products)


@tool
def get_rating(product_id: int) -> str:
    """
    Get the average customer rating and total review count for a product by its ID.
    Returns a JSON object with: product_id, average_rating, review_count.
    """
    result = get_product_rating(product_id)
    return json.dumps(result)


@tool
def checkout(product_id: int) -> str:
    """
    Place an order for the given product ID. Saves the order to the database and returns
    a confirmation message with the order ID, product name, and price.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, price FROM products WHERE id = ?", (product_id,))
    row = cursor.fetchone()

    if not row:
        conn.close()
        return f"Error: product with ID {product_id} not found."

    name, price = row
    cursor.execute(
        "INSERT INTO orders (product_id, product_name, price) VALUES (?, ?, ?)",
        (product_id, name, price),
    )
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return (
        f"Order #{order_id} confirmed! '{name}' has been successfully ordered for ${price:.2f}. "
        f"Your order will arrive in 3-5 business days. Thank you for shopping with us!"
    )


class ImageAttributes(BaseModel):
    is_product: bool = Field(description="True for recognizable goods or food ingredients, including loose/unpackaged oats, grains, honey, fruit, and nuts. Packaging, branding, and labels are NOT required. False for live animals, people, scenery, or unidentifiable subjects.")
    product_type: str = Field(description="What is actually visible, including non-product subjects")
    search_query: Optional[str] = Field(default=None, description="Short ordinary words naming the visible item, e.g. oats or honey; no brand, packaging, underscores, or descriptive scene words. Null for non-products or uncertainty.")
    is_organic: Optional[bool] = Field(default=None, description="True or false only with visible label evidence; otherwise null")
    description: str


image_parser = PydanticOutputParser(pydantic_object=ImageAttributes)


@tool
def describe_product_image(image_path: str) -> str:
    """
    Analyze a product image and return its key attributes as a JSON object.
    Use this when the user uploads a photo of a product they are interested in.
    The returned attributes can be used directly with search_products.
    """
    # Decode locally and send a consistent format, regardless of file extension.
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = "image/jpeg"

    message = HumanMessage(content=[
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_data}"},
        },
        {
            "type": "text",
            "text": (
                "Identify the main visible subject independently of any store inventory. "
                "Recognizable food ingredients count as products even when loose, in a bowl, "
                "or without packaging, branding, price tags, or labels. "
                "For a bowl of oat flakes: is_product=true, product_type=rolled oats, "
                "search_query=oats, is_organic=null. For honey in a jar without a label: "
                "is_product=true, search_query=honey, is_organic=null. "
                "A live animal (for example an elephant), person, scenery, or unclear image is not a product: "
                "set is_product=false and search_query=null. Never substitute a grocery item. "
                "Do not infer organic status without a visible label.\n"
                + image_parser.get_format_instructions()
            ),
        },
    ])

    response = vision_llm.invoke([message])
    return image_parser.parse(response.text).model_dump_json()


def find_products_by_image(image_path: str) -> str:
    """Analyze one upload independently and render only actual database matches."""
    attributes = ImageAttributes.model_validate_json(
        describe_product_image.invoke({"image_path": image_path})
    )
    query = " ".join((attributes.search_query or "").replace("_", " ").split())
    if not attributes.is_product or not query or "%" in query:
        return "I couldn't identify a store product in this image. Please upload a clear product photo or describe what you want."
    products = json.loads(search_products.invoke({
        "query": query, "is_organic": attributes.is_organic,
    }))
    if not products:
        return "No matching products were found in our store for this image. Please try another product photo or describe what you want."
    lines = ["Matching products from our store:"]
    for number, product in enumerate(products, 1):
        rating = json.loads(get_rating.invoke({"product_id": product["id"]}))
        stars = rating.get("average_rating")
        rating_text = f"{stars:.2f}" if stars is not None else "unrated"
        organic = "organic" if product["is_organic"] else "non-organic"
        lines.append(
            f"#{number}. {product['name']} (ID:{product['id']}) — "
            f"$ {product['price']:.2f} ★{rating_text} — {organic}"
        )
    lines.append("Would you like to order one? Tell me its number.")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

agent = create_agent(
    tools=[search_products, get_rating, checkout, describe_product_image],
    model=llm,
    system_prompt=(
        "You are a helpful shopping assistant. Follow these rules strictly.\n\n"
        "IMAGE SEARCH — when the user provides an image path:\n"
        "1. Call describe_product_image with the path to identify the product.\n"
        "2. Use the returned search_query and is_organic to call search_products.\n"
        "3. Continue with the BROWSING flow from step 2 onwards.\n\n"
        "BROWSING — when the user describes what they want to buy:\n"
        "1. Call search_products to find matching items (apply any price/organic filters given).\n"
        "2. For each candidate, call get_rating to retrieve its average rating.\n"
        "3. Filter by the user's minimum rating if specified.\n"
        "4. Present qualifying products as a numbered list. For each item use this exact format "
        "   (plain text, no backticks, no code blocks, no bold, no italic):\n\n"
        "   #<number>. <name> (ID:<product_id>) — $<price> ★<rating> — <organic or non-organic>\n\n"
        "   Add a blank line between each product entry for readability. "
        "   Always include (ID:X) so you can reference it later.\n"
        "5. If only one product qualifies, still show it in the list and ask: "
        "   'Would you like to order it? Just say yes or give me the number.'\n"
        "6. Do NOT call checkout at this stage.\n\n"
        "ORDERING — when the user confirms they want to buy (e.g. 'yes', 'sure', 'go ahead', "
        "'order number 2', 'the first one', 'get me #3'):\n"
        "1. Look at your previous message to find the (ID:X) for the chosen product "
        "   (if only one was listed and the user says 'yes', use that product's ID).\n"
        "2. Call checkout with that product_id (the number from (ID:X)).\n"
        "3. Confirm the order to the user in plain text.\n\n"
        "Never place an order unless the user explicitly confirms. "
        "Never guess a product_id — always take it from the (ID:X) in your own previous message."
    ),
)

if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "I want to buy organic honey with 4.5+ rating and less than $20 price."
                    ),
                }
            ]
        }
    )
    print(result["messages"][-1].content)
