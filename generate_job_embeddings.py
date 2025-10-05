import sqlite3
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.resume_service import ResumeParsingService

def generate_job_embeddings():
    try:
        # Initialize resume service
        resume_service = ResumeParsingService()
        
        # Connect to database
        conn = sqlite3.connect('resumerag.db')
        cursor = conn.cursor()
        
        # Get all jobs without embeddings
        cursor.execute('SELECT id, title, description, requirements, company FROM jobs WHERE embedding IS NULL OR embedding = ""')
        jobs = cursor.fetchall()
        
        print(f"Found {len(jobs)} jobs needing embeddings...")
        
        for job_id, title, description, requirements_json, company in jobs:
            try:
                # Parse requirements
                requirements = json.loads(requirements_json) if requirements_json else []
                
                # Create job text for embedding
                job_text = f"{title} at {company}. {description}"
                if requirements:
                    job_text += f" Requirements: {', '.join(requirements)}"
                
                # Generate embedding
                embedding = resume_service.create_embedding(job_text)
                
                # Update job with embedding
                cursor.execute(
                    'UPDATE jobs SET embedding = ? WHERE id = ?',
                    (json.dumps(embedding), job_id)
                )
                
                print(f"Generated embedding for: {title} at {company}")
                
            except Exception as e:
                print(f"Error processing job {job_id}: {e}")
                continue
        
        conn.commit()
        print(f"\nSuccessfully generated embeddings for {len(jobs)} jobs!")
        
        # Verify embeddings were created
        cursor.execute('SELECT COUNT(*) FROM jobs WHERE embedding IS NOT NULL AND embedding != ""')
        count = cursor.fetchone()[0]
        print(f"Total jobs with embeddings: {count}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error generating job embeddings: {e}")

if __name__ == "__main__":
    generate_job_embeddings()
