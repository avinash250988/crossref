# FastAPI application for Dual Database Product Catalog Search API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Union
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import logging
import boto3
import os
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# S3 Configuration
S3_BUCKET_NAME = "product-matching-model-bucket"
S3_REGION = "us-east-1"

def download_from_s3(bucket_name: str, s3_key: str, local_filename: str) -> bool:
    """Download a file from S3 if it doesn't exist locally"""
    try:
        if os.path.exists(local_filename):
            logger.info(f"{local_filename} already exists locally, skipping download")
            return True
        
        logger.info(f"Downloading {s3_key} from S3 bucket {bucket_name}...")
        s3_client = boto3.client('s3', region_name=S3_REGION)
        s3_client.download_file(bucket_name, s3_key, local_filename)
        logger.info(f"Successfully downloaded {local_filename}")
        return True
        
    except ClientError as e:
        logger.error(f"Failed to download {s3_key} from S3: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {s3_key}: {e}")
        return False

def ensure_data_files():
    """Ensure all required data files are available locally"""
    files_to_download = [
        {"s3_key": "450_embeddings.pkl", "local_file": "450_embeddings.pkl"},
        {"s3_key": "catalog_embeddings3.pkl", "local_file": "catalog_embeddings3.pkl"}
    ]
    
    for file_info in files_to_download:
        success = download_from_s3(S3_BUCKET_NAME, file_info["s3_key"], file_info["local_file"])
        if not success:
            raise Exception(f"Failed to download required file: {file_info['local_file']}")
    
    logger.info("All required data files are available")

# Initialize FastAPI app
app = FastAPI(
    title="Dual Database Product Catalog Search API",
    description="API for searching products using semantic similarity across two databases and returning AI-generated matches",
    version="2.0.0"
)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    # Common fields
    item_description: str
    mpc: Optional[str] = None
    supplier_no: Optional[str] = None
    
    # 450 Transformer DB specific fields
    upc: Optional[str] = None
    gtin_450: Optional[str] = None
    unique_id: Optional[str] = None  # New field for 450 database
    
    # Product Catalog DB specific fields
    brand_name: Optional[str] = None
    pack_size: Optional[str] = None
    
    # Allow any additional fields to be passed through
    class Config:
        extra = "allow"
        max_extra_fields = 60

# Union type to handle both single request and list of requests
BatchSearchRequest = Union[SearchRequest, List[SearchRequest]]

class ProductResult(BaseModel):
    # Input fields (preserved)
    item_description: str
    mpc: str
    supplier_no: str
    
    # 450 Transformer DB specific input fields
    upc: Optional[str] = None
    gtin_450: Optional[str] = None
    unique_id: Optional[str] = None  # Input UniqueID for 450 database
    
    # Product Catalog DB specific input fields
    brand_name: Optional[str] = None
    pack_size: Optional[str] = None
    
    # AI-generated output fields (prefixed with AI_)
    AI_item_description: str
    AI_GTIN: int
    AI_MPC: str
    AI_supplier_no: int
    AI_upc: Optional[int] = None
    AI_gtin_450: Optional[int] = None
    AI_unique_id: Optional[str] = None  # Output UniqueID from 450 database
    AI_brand_name: Optional[str] = None
    AI_pack_size: Optional[str] = None
    
    # Search metadata
    matching_score: float
    source: str

class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: List[ProductResult]
    additional_fields: dict = {}

# Union type to handle both single response and list of responses
BatchSearchResponse = Union[SearchResponse, List[SearchResponse]]

# Global variables for models and databases
model = None
vector_db_450 = None
vector_db_product = None

def load_model():
    """Load the sentence transformer model"""
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    return model

def load_450_database():
    """Load the 450 transformer database"""
    global vector_db_450
    if vector_db_450 is None:
        ensure_data_files()  # Download from S3 if needed
        with open('450_embeddings.pkl', 'rb') as f:
            vector_db_450 = pickle.load(f)
    return vector_db_450

def load_product_database():
    """Load the product catalog database"""
    global vector_db_product
    if vector_db_product is None:
        ensure_data_files()  # Download from S3 if needed
        with open('catalog_embeddings3.pkl', 'rb') as f:
            vector_db_product = pickle.load(f)
    return vector_db_product

def search_450_database(query: str, upc_filter: Optional[str] = None, 
                       mpc_filter: Optional[str] = None, gtin_450_filter: Optional[str] = None, 
                       supplier_filter: Optional[str] = None, unique_id_filter: Optional[str] = None,
                       brand_filter: Optional[str] = None, pack_size_filter: Optional[str] = None) -> Optional[dict]:
    """Search in 450 Transformer database"""
    try:
        vector_db = load_450_database()
        model = load_model()
        
        descriptions = vector_db['descriptions']
        gtins = vector_db['gtins']
        supplier_numbers = vector_db['supplier_numbers']
        mpcs = vector_db['mpcs']
        upcs = vector_db['upcs']
        gtin_450s = vector_db['gtin_450s']
        unique_ids = vector_db.get('unique_ids', [])  # New UniqueID field
        embeddings = vector_db['embeddings']
        
        # Create embedding for the query
        query_embedding = model.encode([query])
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Find best match with filters
        best_match = None
        best_score = 0
        matches_found = 0
        
        
        for i, similarity in enumerate(similarities):
            # Apply optional filters - skip if filter is provided but doesn't match
            skip_record = False
            
            # Check UPC filter (only if provided and valid)
            if upc_filter and upc_filter.strip() and upc_filter.strip().lower() != "string":
                if str(upcs[i]).strip() != str(upc_filter).strip():
                    skip_record = True
            
            # Check MPC filter (only if provided and valid)
            if mpc_filter and mpc_filter.strip() and mpc_filter.strip().lower() != "string":
                if str(mpcs[i]).strip() != str(mpc_filter).strip():
                    skip_record = True
            
            # Check GTIN_450 filter (only if provided and valid)
            if gtin_450_filter and gtin_450_filter.strip() and gtin_450_filter.strip().lower() != "string":
                if str(gtin_450s[i]).strip() != str(gtin_450_filter).strip():
                    skip_record = True
            
            # Check supplier filter (only if provided and valid)
            if supplier_filter and supplier_filter.strip() and supplier_filter.strip().lower() != "string":
                db_supplier = str(supplier_numbers[i]).strip() if supplier_numbers[i] else ""
                filter_supplier = str(supplier_filter).strip()
                supplier_match = False
                if db_supplier == filter_supplier:
                    supplier_match = True
                else:
                    try:
                        if float(db_supplier) == float(filter_supplier):
                            supplier_match = True
                    except (ValueError, TypeError):
                        pass
                
                if not supplier_match:
                    skip_record = True
            
            # Skip this record if any provided filter doesn't match
            if skip_record:
                continue
            
            # Track matches (only records that pass all provided filters)
            matches_found += 1
            
            if similarity > best_score:
                best_score = similarity
                best_match = i
        
        if best_match is not None:
            # Safe conversion to integers
            try:
                gtin_val = int(gtins[best_match]) if gtins[best_match] and str(gtins[best_match]).strip() else 0
                supplier_val = int(supplier_numbers[best_match]) if supplier_numbers[best_match] and str(supplier_numbers[best_match]).strip() else 0
                mpc_val = str(mpcs[best_match]) if mpcs[best_match] and str(mpcs[best_match]).strip() else ""
                upc_val = int(upcs[best_match]) if upcs[best_match] and str(upcs[best_match]).strip() else None
                gtin_450_val = int(gtin_450s[best_match]) if gtin_450s[best_match] and str(gtin_450s[best_match]).strip() else None
                # Extract UniqueID from the matched record
                unique_id_val = str(unique_ids[best_match]) if best_match < len(unique_ids) and unique_ids[best_match] else None
            except (ValueError, TypeError):
                gtin_val = 0
                supplier_val = 0
                mpc_val = ""
                upc_val = None
                gtin_450_val = None
                unique_id_val = None
            
            return {
                'item_description': query,
                'mpc': mpc_filter,  # Always show input value
                'supplier_no': supplier_filter,  # Always show input value
                'upc': upc_filter,  # Always show input value
                'gtin_450': gtin_450_filter,  # Always show input value
                'unique_id': unique_id_filter,  # Always show input value
                'brand_name': brand_filter,  # Always show input value
                'pack_size': pack_size_filter,  # Always show input value
                'AI_item_description': descriptions[best_match],  # Always show database value
                'AI_GTIN': gtin_val,  # Always show database value
                'AI_MPC': mpc_val,  # Always show database value
                'AI_supplier_no': supplier_val,  # Always show database value
                'AI_upc': upc_val,  # Always show database value
                'AI_gtin_450': gtin_450_val,  # Always show database value
                'AI_unique_id': unique_id_val,  # Always show database value
                'AI_brand_name': None,  # 450 database doesn't have brand_name
                'AI_pack_size': None,  # 450 database doesn't have pack_size
                'matching_score': round(float(best_score), 2),
                'source': '450_transformer_master_data'
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error searching 450 database: {e}")
        return None

def search_product_database(query: str, mpc_filter: Optional[str] = None, 
                           supplier_filter: Optional[str] = None, brand_filter: Optional[str] = None, 
                           pack_size_filter: Optional[str] = None, upc_filter: Optional[str] = None,
                           gtin_450_filter: Optional[str] = None, unique_id_filter: Optional[str] = None) -> Optional[dict]:
    """Search in Product Catalog database"""
    try:
        vector_db = load_product_database()
        model = load_model()
        
        descriptions = vector_db['descriptions']
        brand_names = vector_db['brand_names']
        gtins = vector_db['gtins']
        supplier_numbers = vector_db['supplier_numbers']
        mpcs = vector_db['mpcs']
        upcs = vector_db['upcs']
        gtin_450s = vector_db['gtin_450s']
        pack_sizes = vector_db['pack_sizes']
        desc_embeddings = vector_db['desc_embeddings']
        brand_embeddings = vector_db['brand_embeddings']
        pack_size_embeddings = vector_db['pack_embeddings']
        
        # Create embeddings for the query
        query_embedding = model.encode([query])
        brand_query_embedding = model.encode([brand_filter]) if brand_filter else None
        
        # Calculate similarities for descriptions
        desc_similarities = cosine_similarity(query_embedding, desc_embeddings)[0]
        
        # Calculate similarities for brand names if provided
        brand_similarities = None
        if brand_query_embedding is not None:
            brand_similarities = cosine_similarity(brand_query_embedding, brand_embeddings)[0]
        
        # Find best match with filters
        best_match = None
        best_score = 0
        matches_found = 0
        
        
        for i, desc_similarity in enumerate(desc_similarities):
            # Apply optional filters - skip if filter is provided but doesn't match
            skip_record = False
            
            # Check MPC filter (only if provided and valid)
            if mpc_filter and mpc_filter.strip() and mpc_filter.strip().lower() != "string":
                if str(mpcs[i]).strip() != str(mpc_filter).strip():
                    skip_record = True
            
            # Check supplier filter (only if provided and valid)
            if supplier_filter and supplier_filter.strip() and supplier_filter.strip().lower() != "string":
                db_supplier = str(supplier_numbers[i]).strip() if supplier_numbers[i] else ""
                filter_supplier = str(supplier_filter).strip()
                supplier_match = False
                if db_supplier == filter_supplier:
                    supplier_match = True
                else:
                    try:
                        if float(db_supplier) == float(filter_supplier):
                            supplier_match = True
                    except (ValueError, TypeError):
                        pass
                
                if not supplier_match:
                    skip_record = True
            
            # Check brand filter (only if provided and valid)
            if brand_filter and brand_filter.strip() and brand_filter.strip().lower() != "string":
                if str(brand_names[i]).strip().lower() != str(brand_filter).strip().lower():
                    skip_record = True
            
            # Check pack size filter (only if provided and valid)
            if pack_size_filter and pack_size_filter.strip() and pack_size_filter.strip().lower() != "string":
                if str(pack_sizes[i]).strip().lower() != str(pack_size_filter).strip().lower():
                    skip_record = True
            
            # Skip this record if any provided filter doesn't match
            if skip_record:
                continue
            
            # Use only description similarity as the score
            similarity_score = desc_similarity
            
            # Track matches (only records that pass all provided filters)
            matches_found += 1
            
            if similarity_score > best_score:
                best_score = similarity_score
                best_match = i
        
        if best_match is not None:
            # Safe conversion to integers
            try:
                gtin_val = int(gtins[best_match]) if gtins[best_match] and str(gtins[best_match]).strip() else 0
                supplier_val = int(supplier_numbers[best_match]) if supplier_numbers[best_match] and str(supplier_numbers[best_match]).strip() else 0
                mpc_val = str(mpcs[best_match]) if mpcs[best_match] and str(mpcs[best_match]).strip() else ""
                brand_val = str(brand_names[best_match]) if brand_names[best_match] and str(brand_names[best_match]).strip() else ""
                pack_size_val = str(pack_sizes[best_match]) if pack_sizes[best_match] and str(pack_sizes[best_match]).strip() else ""
            except (ValueError, TypeError):
                gtin_val = 0
                supplier_val = 0
                mpc_val = ""
                brand_val = ""
                pack_size_val = ""
            
            return {
                'item_description': query,
                'mpc': mpc_filter,  # Always show input value
                'supplier_no': supplier_filter,  # Always show input value
                'upc': upc_filter,  # Always show input value
                'gtin_450': gtin_450_filter,  # Always show input value
                'unique_id': unique_id_filter,  # Always show input value
                'brand_name': brand_filter,  # Always show input value
                'pack_size': pack_size_filter,  # Always show input value
                'AI_item_description': descriptions[best_match],  # Always show database value
                'AI_GTIN': gtin_val,  # Always show database value
                'AI_MPC': mpc_val,  # Always show database value
                'AI_supplier_no': supplier_val,  # Always show database value
                'AI_upc': None,  # Product catalog doesn't have UPC
                'AI_gtin_450': None,  # Product catalog doesn't have GTIN_450
                'AI_brand_name': brand_val,  # Always show database value
                'AI_pack_size': pack_size_val,  # Always show database value
                'matching_score': round(float(best_score), 2),
                'source': 'product_catalog_master_data'
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Error searching product database: {e}")
        return None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Dual Database Product Catalog Search API",
        "version": "2.0.0",
        "endpoints": {
            "/search": "POST - Search both databases",
            "/health": "GET - Health check",
            "/stats": "GET - Database statistics"
        }
    }

async def process_single_search(request: SearchRequest) -> SearchResponse:
    """
    Process a single search request
    """
    # Validate query
    if not request.item_description.strip():
        raise HTTPException(status_code=400, detail="Item description cannot be empty")
    
    results = []
    
    # Search 450 Transformer database
    result_450 = search_450_database(
        query=request.item_description,
        upc_filter=request.upc,
        mpc_filter=request.mpc,
        gtin_450_filter=request.gtin_450,
        supplier_filter=request.supplier_no,
        unique_id_filter=request.unique_id,
        brand_filter=request.brand_name,
        pack_size_filter=request.pack_size
    )
    if result_450:
        results.append(result_450)
    
    # Search Product Catalog database
    result_product = search_product_database(
        query=request.item_description,
        mpc_filter=request.mpc,
        supplier_filter=request.supplier_no,
        brand_filter=request.brand_name,
        pack_size_filter=request.pack_size,
        upc_filter=request.upc,
        gtin_450_filter=request.gtin_450,
        unique_id_filter=request.unique_id
    )
    if result_product:
        results.append(result_product)
    
    # Capture any additional fields that were sent in the request
    additional_fields = {}
    for field_name, field_value in request.model_dump().items():
        if field_name not in ['item_description', 'mpc', 'supplier_no', 'upc', 'gtin_450', 'unique_id', 'brand_name', 'pack_size']:
            additional_fields[field_name] = field_value
    
    return SearchResponse(
        query=request.item_description,
        total_results=len(results),
        results=results,
        additional_fields=additional_fields
    )

@app.post("/search", response_model=BatchSearchResponse)
async def search_endpoint(request: BatchSearchRequest):
    """
    Search for products using semantic similarity across both databases
    
    Accepts:
    - Single search request: SearchRequest object
    - Batch search requests: List[SearchRequest]
    
    Returns:
    - Single response: SearchResponse object
    - Batch responses: List[SearchResponse]
    
    Each result contains:
    - AI_GTIN, AI_MPC, AI_supplier_no, AI_unique_id fields for each product
    - Similarity match score field for each product
    - Source identification for each result
    """
    try:
        # Handle single request
        if isinstance(request, dict) or hasattr(request, 'item_description'):
            logger.info("Processing single search request")
            return await process_single_search(request)
        
        # Handle batch requests (list)
        elif isinstance(request, list):
            # Add batch size limit for performance
            MAX_BATCH_SIZE = 50
            if len(request) > MAX_BATCH_SIZE:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Batch size too large. Maximum {MAX_BATCH_SIZE} records allowed per request. "
                           f"Please split your {len(request)} records into smaller batches."
                )
            
            logger.info(f"Processing batch search request with {len(request)} items")
            
            # Pre-load models and databases once for the entire batch
            model = load_model()
            vector_db_450 = load_450_database()
            vector_db_product = load_product_database()
            logger.info("Models and databases pre-loaded for batch processing")
            
            batch_results = []
            
            for i, single_request in enumerate(request):
                try:
                    if i % 10 == 0:  # Log progress every 10 items
                        logger.info(f"Processing batch item {i+1}/{len(request)}")
                    
                    result = await process_single_search(single_request)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing batch item {i}: {str(e)}")
                    # Continue processing other items, but log the error
                    error_response = SearchResponse(
                        query=getattr(single_request, 'item_description', 'unknown'),
                        total_results=0,
                        results=[],
                        additional_fields={"error": str(e)}
                    )
                    batch_results.append(error_response)
            
            logger.info(f"Batch processing completed: {len(batch_results)} results")
            return batch_results
        
        else:
            raise HTTPException(status_code=400, detail="Invalid request format. Expected single SearchRequest or List[SearchRequest]")
            
    except Exception as e:
        logger.error(f"Search endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test if models and databases can be loaded
        load_model()
        load_450_database()
        load_product_database()
        return {"status": "healthy", "message": "API is running successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        vector_db_450 = load_450_database()
        vector_db_product = load_product_database()
        return {
            "450_transformer_db": {
                "total_products": len(vector_db_450['descriptions']),
                "fields": list(vector_db_450.keys())
            },
            "product_catalog_db": {
                "total_products": len(vector_db_product['descriptions']),
                "fields": list(vector_db_product.keys())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)
