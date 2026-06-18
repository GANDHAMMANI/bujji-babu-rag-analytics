# test_query.py
from ingestion import ingest_pdf, get_metadata
from query import query_pdf

pdf_id = "6c5da9257e"  # check extracted/ folder name
from ingestion import _metadata
print("Images in metadata:", _metadata.get(pdf_id, {}).get('image_count'))

result = query_pdf(pdf_id, "what is deep learning")
print("Images in result:", len(result.get('images', [])))
print("Image paths:", [i['path'] for i in result.get('images', [])])