# Product Catalog Search API

This is a FastAPI-based application that provides an API for searching through a product catalog using semantic similarity. The application receives data through API calls and returns GTIN fields and similarity match scores.

## Features

- **Semantic Search**: Uses AI-powered semantic similarity to find products based on descriptions
- **GTIN Field**: Returns the GTIN (Global Trade Item Number) for each product
- **Similarity Match Score**: Provides a similarity score (0-1) indicating how well each product matches the query
- **Filtering Options**: Support for filtering by supplier number, MPC, UPC, and 450 GTIN
- **RESTful API**: Clean, documented API endpoints

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure your AWS credentials to access the S3 bucket containing the PKL files.

## S3 Configuration

The application automatically downloads required PKL files from S3:
- **Bucket**: `product-matching-model-bucket`
- **Region**: `us-east-1`
- **Files**: `450_embeddings.pkl` and `catalog_embeddings3.pkl`

Make sure your AWS credentials have read access to this S3 bucket.

## Running the API

Start the API server:
```bash
python app_fixed6_prod_ready.py
```

The API will be available at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
- **GET** `/`
- Returns API information and available endpoints

### 2. Search Products
- **POST** `/search`
- Main endpoint for searching products

**Request Body:**
```json
{
    "query": "mayo salad dressing",
    "supplier_filter": "1",
    "mpc_filter": "10952CLG",
    "upc_filter": "10071314049723",
    "gtin_450_filter": "4501234567890",
    "top_k": 5
}
```

**Response:**
```json
{
    "query": "mayo salad dressing",
    "total_results": 3,
    "results": [
        {
            "description": "Mayonnaise Salad Dressing",
            "gtin": "1234567890123",
            "supplier_number": "1",
            "mpc": "10952CLG",
            "upc": "10071314049723",
            "gtin_450": "4501234567890",
            "similarity_score": 0.85
        }
    ],
    "filters_applied": {
        "supplier_filter": "1"
    }
}
```

### 3. Health Check
- **GET** `/health`
- Returns API health status

### 4. Database Statistics
- **GET** `/stats`
- Returns information about the product database

## API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **ReDoc documentation**: `http://localhost:8000/redoc`

## Example Usage

### Using curl:
```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "cleaning supplies",
       "top_k": 3
     }'
```

### Using Python requests:
```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={
        "query": "cleaning supplies",
        "top_k": 3
    }
)

results = response.json()
for product in results["results"]:
    print(f"GTIN: {product['gtin']}")
    print(f"Similarity Score: {product['similarity_score']}")
    print(f"Description: {product['description']}")
    print("---")
```

## Key Changes from UI Version

- **Removed**: All Streamlit UI components and styling
- **Added**: FastAPI framework with RESTful endpoints
- **Enhanced**: GTIN field is now prominently returned in API responses
- **New**: Similarity match score field provides confidence metrics
- **Improved**: Better error handling and validation
- **Added**: Health check and statistics endpoints

## Notes

- The similarity score ranges from 0 to 1, where higher values indicate better matches
- All filters are optional and can be combined
- The API automatically handles model loading and caching for better performance
- GTIN values are returned as strings to preserve leading zeros and formatting
