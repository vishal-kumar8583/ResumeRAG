import os
import json
import re
from typing import List, Dict, Tuple, Optional
import PyPDF2
import docx
import random
import hashlib
from collections import Counter
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy
from fuzzywuzzy import fuzz, process
import pickle
import hashlib
from datetime import datetime

class ResumeParsingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Initialize advanced ML models
        self.model_name = model_name
        self.sentence_model = None
        self.nlp = None
        self.tfidf_vectorizer = None
        self.skill_clusters = None
        self.cache_dir = "./cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize models
        self._initialize_models()
        
        # Enhanced PII patterns
        self.pii_patterns = {
            'phone': r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b(?:\d{3}-\d{2}-\d{4}|\d{9})\b',
            'address': r'\d+\s+[A-Za-z\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl)',
            'linkedin': r'linkedin\.com/in/[\w-]+',
            'github': r'github\.com/[\w-]+',
            'portfolio': r'(?:www\.)?[\w-]+\.(?:com|net|org|io)',
        }
        
        # Advanced skill taxonomy
        self.skill_taxonomy = {
            'programming_languages': {
                'python': ['python', 'py', 'django', 'flask', 'fastapi'],
                'javascript': ['javascript', 'js', 'node', 'react', 'angular', 'vue'],
                'java': ['java', 'spring', 'hibernate', 'maven'],
                'csharp': ['c#', 'csharp', '.net', 'asp.net'],
                'cpp': ['c++', 'cpp', 'c plus plus'],
                'go': ['go', 'golang'],
                'rust': ['rust'],
                'php': ['php', 'laravel', 'symfony'],
                'ruby': ['ruby', 'rails'],
                'swift': ['swift', 'ios'],
                'kotlin': ['kotlin', 'android'],
                'typescript': ['typescript', 'ts'],
                'r': ['r', 'r language'],
                'matlab': ['matlab'],
                'sql': ['sql', 'mysql', 'postgresql', 'sqlite'],
            },
            'frameworks': {
                'react': ['react', 'reactjs', 'jsx'],
                'angular': ['angular', 'angularjs'],
                'vue': ['vue', 'vuejs'],
                'django': ['django'],
                'flask': ['flask'],
                'spring': ['spring', 'spring boot'],
                'express': ['express', 'expressjs'],
                'laravel': ['laravel'],
                'rails': ['rails', 'ruby on rails'],
                'fastapi': ['fastapi'],
                'nextjs': ['nextjs', 'next.js'],
                'nuxt': ['nuxt', 'nuxtjs'],
            },
            'databases': {
                'mysql': ['mysql'],
                'postgresql': ['postgresql', 'postgres'],
                'mongodb': ['mongodb', 'mongo'],
                'redis': ['redis'],
                'oracle': ['oracle'],
                'sqlite': ['sqlite'],
                'cassandra': ['cassandra'],
                'elasticsearch': ['elasticsearch', 'elastic'],
                'dynamodb': ['dynamodb'],
                'neo4j': ['neo4j'],
            },
            'cloud_platforms': {
                'aws': ['aws', 'amazon web services', 'ec2', 's3', 'lambda'],
                'azure': ['azure', 'microsoft azure'],
                'gcp': ['gcp', 'google cloud', 'google cloud platform'],
                'docker': ['docker', 'containerization'],
                'kubernetes': ['kubernetes', 'k8s'],
                'terraform': ['terraform'],
                'ansible': ['ansible'],
            },
            'data_science': {
                'machine_learning': ['machine learning', 'ml', 'deep learning', 'neural networks'],
                'tensorflow': ['tensorflow', 'tf'],
                'pytorch': ['pytorch', 'torch'],
                'pandas': ['pandas'],
                'numpy': ['numpy'],
                'scikit': ['scikit-learn', 'sklearn'],
                'opencv': ['opencv'],
                'matplotlib': ['matplotlib'],
                'seaborn': ['seaborn'],
                'jupyter': ['jupyter'],
            }
        }
    
    def _initialize_models(self):
        """Initialize ML models with caching"""
        try:
            # Initialize sentence transformer
            model_path = os.path.join(self.cache_dir, "sentence_model")
            if os.path.exists(model_path):
                self.sentence_model = SentenceTransformer(model_path)
            else:
                self.sentence_model = SentenceTransformer(self.model_name)
                self.sentence_model.save(model_path)
            
            # Initialize spaCy (try to load, fallback to basic if not available)
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("spaCy model not found, using basic NLP")
                self.nlp = None
            
            # Initialize TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            
            print("Advanced ML models initialized successfully")
            
        except Exception as e:
            print(f"Error initializing models: {e}")
            # Fallback to basic models
            self.sentence_model = None
            self.nlp = None
            self.tfidf_vectorizer = None
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting PDF text: {str(e)}")
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting DOCX text: {str(e)}")
    
    def extract_text(self, file_path: str, filename: str) -> str:
        """Extract text based on file extension"""
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext == 'pdf':
            return self.extract_text_from_pdf(file_path)
        elif file_ext in ['docx', 'doc']:
            return self.extract_text_from_docx(file_path)
        elif file_ext == 'txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            raise Exception(f"Unsupported file format: {file_ext}")
    
    def redact_pii(self, text: str) -> str:
        """Redact PII from text"""
        redacted_text = text
        
        # Redact patterns
        for pii_type, pattern in self.pii_patterns.items():
            if pii_type == 'phone':
                redacted_text = re.sub(pattern, '[PHONE_REDACTED]', redacted_text)
            elif pii_type == 'email':
                redacted_text = re.sub(pattern, '[EMAIL_REDACTED]', redacted_text)
            elif pii_type == 'ssn':
                redacted_text = re.sub(pattern, '[SSN_REDACTED]', redacted_text)
            elif pii_type == 'address':
                redacted_text = re.sub(pattern, '[ADDRESS_REDACTED]', redacted_text, flags=re.IGNORECASE)
        
        return redacted_text
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text using advanced taxonomy"""
        text_lower = text.lower()
        found_skills = []
        
        # Extract skills using taxonomy
        for category, skills_dict in self.skill_taxonomy.items():
            for skill_name, variations in skills_dict.items():
                for variation in variations:
                    if variation in text_lower:
                        # Check if it's a standalone skill (not part of another word)
                        if self._is_standalone_skill(text_lower, variation):
                            found_skills.append(skill_name.title())
                            break  # Avoid duplicates for same skill
        
        # Extract additional skills using NLP if available
        if self.nlp:
            additional_skills = self._extract_skills_with_nlp(text)
            found_skills.extend(additional_skills)
        
        # Remove duplicates and return
        return list(set(found_skills))
    
    def _extract_skills_with_nlp(self, text: str) -> List[str]:
        """Extract skills using NLP techniques"""
        additional_skills = []
        
        try:
            doc = self.nlp(text)
            
            # Extract noun phrases that might be skills
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower().strip()
                
                # Skip very short or very long chunks
                if len(chunk_text) < 3 or len(chunk_text) > 30:
                    continue
                
                # Look for technical terms
                technical_indicators = ['technology', 'framework', 'library', 'tool', 'platform', 'language']
                if any(indicator in chunk_text for indicator in technical_indicators):
                    # Extract the main term (usually the first word)
                    main_term = chunk_text.split()[0]
                    if len(main_term) > 2:
                        additional_skills.append(main_term.title())
            
            # Extract skills from "Skills:" sections
            skills_patterns = [
                r'skills?[:\s]+([^.]+)',
                r'technologies?[:\s]+([^.]+)',
                r'expertise[:\s]+([^.]+)',
                r'proficient\s+in[:\s]+([^.]+)',
            ]
            
            for pattern in skills_patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    # Split by common separators
                    skills_in_match = re.split(r'[,;|•\n]', match)
                    for skill in skills_in_match:
                        skill = skill.strip()
                        if len(skill) > 2 and len(skill) < 20:
                            additional_skills.append(skill.title())
            
        except Exception as e:
            print(f"Error in NLP skill extraction: {e}")
        
        return additional_skills[:10]  # Limit additional skills
    
    def extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience from resume text"""
        # Simple pattern matching for experience
        experience_patterns = [
            r'(\d{1,2}/\d{4}|\d{4})\s*-\s*(\d{1,2}/\d{4}|\d{4}|present|current)',
            r'(\w+\s+\d{4})\s*-\s*(\w+\s+\d{4}|present|current)'
        ]
        
        experiences = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) > 10:  # Skip short lines
                for pattern in experience_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Try to extract company and role from surrounding context
                        context_lines = lines[max(0, i-2):min(len(lines), i+3)]
                        context = ' '.join(context_lines)
                        
                        experiences.append({
                            'period': line,
                            'context': context[:200] + '...' if len(context) > 200 else context
                        })
                        break
        
        return experiences[:5]  # Return top 5 experiences
    
    def create_embedding(self, text: str) -> List[float]:
        """Create advanced ML-based embedding using sentence transformers"""
        try:
            if self.sentence_model:
                # Use sentence transformer for high-quality embeddings
                embedding = self.sentence_model.encode(text, convert_to_tensor=False)
                return embedding.tolist()
            else:
                # Fallback to TF-IDF based embedding
                return self._create_tfidf_embedding(text)
                
        except Exception as e:
            print(f"Error creating embedding: {e}")
            # Fallback to simple text-based embedding
            return self._create_simple_embedding(text)
    
    def _create_simple_embedding(self, text: str) -> List[float]:
        """Create simple text-based embedding as ultimate fallback"""
        try:
            words = text.lower().split()
            word_counts = Counter(words)
            
            # Create feature vector based on common keywords
            features = []
            
            # Programming languages
            prog_langs = ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'go', 'rust']
            for lang in prog_langs:
                features.append(float(sum(1 for word in words if lang in word)))
            
            # Technologies
            techs = ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node', 'express', 'aws', 'docker']
            for tech in techs:
                features.append(float(sum(1 for word in words if tech in word)))
            
            # Experience indicators
            exp_words = ['experience', 'years', 'developed', 'built', 'designed', 'managed', 'led', 'achieved']
            for exp in exp_words:
                features.append(float(sum(1 for word in words if exp in word)))
            
            # Education
            edu_words = ['degree', 'university', 'college', 'phd', 'masters', 'bachelor', 'education']
            for edu in edu_words:
                features.append(float(sum(1 for word in words if edu in word)))
            
            # Skills
            skill_words = ['skill', 'proficient', 'expert', 'knowledge', 'familiar', 'certified']
            for skill in skill_words:
                features.append(float(sum(1 for word in words if skill in word)))
            
            # Text statistics
            features.extend([
                float(len(words)),  # total words
                float(len(set(words))),  # unique words
                float(len([w for w in words if len(w) > 6])),  # long words
                float(len([w for w in words if w.isupper()])),  # uppercase words
            ])
            
            # Normalize features to 0-1 range
            max_val = max(features) if features else 1.0
            if max_val > 0:
                features = [f / max_val for f in features]
            
            # Ensure we have at least 50 dimensions
            while len(features) < 50:
                features.append(0.0)
            
            return features[:50]  # Return exactly 50 dimensions
            
        except Exception as e:
            print(f"Error creating simple embedding: {e}")
            return [0.1] * 50
    
    def _create_tfidf_embedding(self, text: str) -> List[float]:
        """Create TF-IDF based embedding as fallback"""
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Create TF-IDF features
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([cleaned_text])
            tfidf_features = tfidf_matrix.toarray()[0]
            
            # Pad or truncate to standard dimension
            target_dim = 384
            if len(tfidf_features) < target_dim:
                # Pad with zeros
                tfidf_features = np.pad(tfidf_features, (0, target_dim - len(tfidf_features)))
            else:
                # Truncate
                tfidf_features = tfidf_features[:target_dim]
            
            return tfidf_features.tolist()
            
        except Exception as e:
            print(f"Error creating TF-IDF embedding: {e}")
            return [0.1] * 384
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better embedding quality"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def similarity_search(self, query_embedding: List[float], resume_embeddings: List[Tuple[int, List[float]]], k: int = 5) -> List[Tuple[int, float]]:
        """Advanced similarity search with multiple algorithms"""
        if not resume_embeddings:
            return []
        
        try:
            # Check if all embeddings have the same dimension
            query_dim = len(query_embedding)
            resume_dims = [len(emb) for _, emb in resume_embeddings]
            
            # If dimensions don't match, use basic cosine similarity
            if not all(dim == query_dim for dim in resume_dims):
                print(f"Dimension mismatch: query={query_dim}, resume_dims={resume_dims}")
                return self._basic_cosine_similarity(query_embedding, resume_embeddings, k)
            
            # Convert to numpy arrays for efficient computation
            query_vec = np.array(query_embedding).reshape(1, -1)
            resume_vecs = np.array([emb for _, emb in resume_embeddings])
            resume_ids = [resume_id for resume_id, _ in resume_embeddings]
            
            # Calculate cosine similarity using sklearn for efficiency
            similarities = cosine_similarity(query_vec, resume_vecs)[0]
            
            # Create list of (resume_id, similarity) tuples
            results = list(zip(resume_ids, similarities))
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:k]
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            # Fallback to basic cosine similarity
            return self._basic_cosine_similarity(query_embedding, resume_embeddings, k)
    
    def _basic_cosine_similarity(self, query_embedding: List[float], resume_embeddings: List[Tuple[int, List[float]]], k: int) -> List[Tuple[int, float]]:
        """Fallback basic cosine similarity calculation"""
        def cosine_similarity(vec1, vec2):
            """Calculate cosine similarity between two vectors"""
            # Handle different dimensions by taking minimum length
            min_len = min(len(vec1), len(vec2))
            if min_len == 0:
                return 0.0
            
            vec1_truncated = vec1[:min_len]
            vec2_truncated = vec2[:min_len]
            
            dot_product = sum(a * b for a, b in zip(vec1_truncated, vec2_truncated))
            norm_a = sum(a * a for a in vec1_truncated) ** 0.5
            norm_b = sum(b * b for b in vec2_truncated) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot_product / (norm_a * norm_b)
        
        similarities = []
        for resume_id, resume_embedding in resume_embeddings:
            sim = cosine_similarity(query_embedding, resume_embedding)
            similarities.append((resume_id, sim))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def advanced_semantic_search(self, query: str, resume_data: List[Dict], k: int = 5) -> List[Dict]:
        """Advanced semantic search combining multiple approaches"""
        if not resume_data:
            return []
        
        try:
            # Create query embedding
            query_embedding = self.create_embedding(query)
            
            # Prepare resume embeddings
            resume_embeddings = []
            for resume in resume_data:
                if 'embedding' in resume:
                    embedding = json.loads(resume['embedding']) if isinstance(resume['embedding'], str) else resume['embedding']
                    resume_embeddings.append((resume['id'], embedding))
            
            # Get semantic similarity scores
            semantic_scores = self.similarity_search(query_embedding, resume_embeddings, k)
            
            # Enhance with fuzzy matching for better recall
            fuzzy_scores = self._calculate_fuzzy_scores(query, resume_data)

            # Add TF-IDF reranking as a classical IR signal (robust for keyword queries)
            tfidf_scores = self._tfidf_rerank_scores(query, resume_data)
            
            # Combine scores using weighted average
            combined_results = []
            for resume_id, semantic_score in semantic_scores:
                fuzzy_score = fuzzy_scores.get(resume_id, 0.0)
                tfidf_score = tfidf_scores.get(resume_id, 0.0)

                # Normalize and fuse: semantic (0.6) + tfidf (0.25) + fuzzy (0.15)
                combined_score = 0.6 * float(semantic_score) + 0.25 * float(tfidf_score) + 0.15 * float(fuzzy_score)
                
                # Find resume data
                resume_info = next((r for r in resume_data if r['id'] == resume_id), None)
                if resume_info:
                    combined_results.append({
                        'resume_id': resume_id,
                        'semantic_score': semantic_score,
                        'fuzzy_score': fuzzy_score,
                        'tfidf_score': tfidf_score,
                        'combined_score': combined_score,
                        'resume_info': resume_info
                    })
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            return combined_results[:k]
            
        except Exception as e:
            print(f"Error in advanced semantic search: {e}")
            return []

    def _tfidf_rerank_scores(self, query: str, resume_data: List[Dict]) -> Dict[int, float]:
        """Compute TF-IDF cosine similarity scores between the query and each resume.
        Returns a mapping of resume_id -> score in [0,1].
        """
        try:
            # Collect corpus (resume contents)
            corpus = []
            ids = []
            for r in resume_data:
                text = r.get('content') or ''
                corpus.append(self._preprocess_text(text))
                ids.append(r['id'])

            if not corpus:
                return {}

            # Fit a lightweight vectorizer (separate from model init to use resume corpus)
            vectorizer = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 2))
            doc_matrix = vectorizer.fit_transform(corpus)

            # Transform query and compute cosine similarities
            q_vec = vectorizer.transform([self._preprocess_text(query)])
            sims = cosine_similarity(q_vec, doc_matrix)[0]

            # Normalize to [0,1]
            if sims.size == 0:
                return {}
            max_sim = float(sims.max()) or 1.0
            scores = {ids[i]: float(sims[i] / max_sim) for i in range(len(ids))}
            return scores
        except Exception as e:
            print(f"Error computing TF-IDF scores: {e}")
            return {}
    
    def _calculate_fuzzy_scores(self, query: str, resume_data: List[Dict]) -> Dict[int, float]:
        """Calculate fuzzy matching scores for better recall"""
        fuzzy_scores = {}
        
        try:
            query_lower = query.lower()
            
            for resume in resume_data:
                resume_id = resume['id']
                content = resume.get('content', '')
                
                # Calculate fuzzy scores for different parts
                scores = []
                
                # Title/name matching
                if 'filename' in resume:
                    filename_score = fuzz.partial_ratio(query_lower, resume['filename'].lower()) / 100.0
                    scores.append(filename_score)
                
                # Content matching
                content_score = fuzz.partial_ratio(query_lower, content.lower()) / 100.0
                scores.append(content_score)
                
                # Extract key phrases and match
                if self.nlp:
                    doc = self.nlp(content)
                    key_phrases = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) <= 3]
                    
                    phrase_scores = []
                    for phrase in key_phrases[:10]:  # Limit to top 10 phrases
                        phrase_score = fuzz.partial_ratio(query_lower, phrase) / 100.0
                        phrase_scores.append(phrase_score)
                    
                    if phrase_scores:
                        scores.append(max(phrase_scores))
                
                # Take the maximum score
                fuzzy_scores[resume_id] = max(scores) if scores else 0.0
                
        except Exception as e:
            print(f"Error calculating fuzzy scores: {e}")
        
        return fuzzy_scores
    
    def _extract_relevant_snippets(self, content: str, query: str, max_snippets: int = 3) -> List[str]:
        """Extract relevant snippets from content based on query"""
        try:
            sentences = content.split('.')
            query_words = query.lower().split()
            relevant_snippets = []
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    sentence_lower = sentence.lower()
                    
                    # Calculate relevance score
                    word_matches = sum(1 for word in query_words if word in sentence_lower)
                    relevance_score = word_matches / len(query_words) if query_words else 0
                    
                    if relevance_score > 0.3:  # At least 30% word match
                        relevant_snippets.append({
                            'sentence': sentence + '.',
                            'score': relevance_score
                        })
            
            # Sort by relevance and return top snippets
            relevant_snippets.sort(key=lambda x: x['score'], reverse=True)
            return [snippet['sentence'] for snippet in relevant_snippets[:max_snippets]]
            
        except Exception as e:
            print(f"Error extracting snippets: {e}")
            return []
    
    def _fuzzy_search_only(self, query: str, resume_data: List[Dict], k: int = 5) -> List[Dict]:
        """Fuzzy search only implementation"""
        try:
            fuzzy_scores = self._calculate_fuzzy_scores(query, resume_data)
            
            # Sort by fuzzy score
            results = []
            for resume_id, score in fuzzy_scores.items():
                resume_info = next((r for r in resume_data if r['id'] == resume_id), None)
                if resume_info:
                    results.append({
                        'resume_id': resume_id,
                        'fuzzy_score': score,
                        'semantic_score': 0.0,
                        'combined_score': score,
                        'resume_info': resume_info
                    })
            
            results.sort(key=lambda x: x['fuzzy_score'], reverse=True)
            return results[:k]
            
        except Exception as e:
            print(f"Error in fuzzy search: {e}")
            return []
    
    def _generate_search_suggestions(self, query: str, resumes: List) -> List[str]:
        """Generate search suggestions based on resume content"""
        suggestions = set()
        query_lower = query.lower()
        
        try:
            # Extract skills from all resumes
            all_skills = set()
            for resume in resumes:
                skills = self.extract_skills(resume.content)
                all_skills.update(skills)
            
            # Find skills that match the query
            for skill in all_skills:
                if query_lower in skill.lower():
                    suggestions.add(skill)
            
            # Extract key phrases using NLP
            if self.nlp:
                for resume in resumes[:3]:  # Limit to first 3 resumes for performance
                    doc = self.nlp(resume.content)
                    for chunk in doc.noun_chunks:
                        chunk_text = chunk.text.strip()
                        if (len(chunk_text) > 3 and len(chunk_text) < 20 and 
                            query_lower in chunk_text.lower()):
                            suggestions.add(chunk_text)
            
            # Convert to list and limit
            suggestions_list = list(suggestions)[:10]
            return suggestions_list
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return []
    
    def _analyze_content_quality(self, content: str) -> Dict[str, any]:
        """Analyze resume content quality"""
        try:
            words = content.split()
            sentences = content.split('.')
            
            # Basic metrics
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            
            # Check for common issues
            issues = []
            suggestions = []
            
            # Length analysis
            if word_count < 200:
                issues.append("Resume is too short")
                suggestions.append("Add more details about your experience and achievements")
            elif word_count > 1000:
                issues.append("Resume is too long")
                suggestions.append("Consider condensing content to focus on most relevant experience")
            
            # Sentence length analysis
            if avg_sentence_length > 25:
                issues.append("Sentences are too long")
                suggestions.append("Break down long sentences for better readability")
            elif avg_sentence_length < 8:
                issues.append("Sentences are too short")
                suggestions.append("Combine related short sentences for better flow")
            
            # Check for action verbs
            action_verbs = ['developed', 'created', 'built', 'designed', 'implemented', 'managed', 'led', 'achieved', 'improved', 'increased']
            action_verb_count = sum(1 for word in words if word.lower() in action_verbs)
            action_verb_ratio = action_verb_count / word_count if word_count > 0 else 0
            
            if action_verb_ratio < 0.02:
                issues.append("Limited use of action verbs")
                suggestions.append("Use more action verbs to describe your achievements")
            
            # Check for quantified achievements
            numbers = re.findall(r'\b\d+\b', content)
            quantified_sentences = sum(1 for sentence in sentences if any(num in sentence for num in numbers))
            quantification_ratio = quantified_sentences / sentence_count if sentence_count > 0 else 0
            
            if quantification_ratio < 0.3:
                issues.append("Limited quantified achievements")
                suggestions.append("Add more numbers and metrics to quantify your achievements")
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 2),
                "action_verb_ratio": round(action_verb_ratio, 3),
                "quantification_ratio": round(quantification_ratio, 3),
                "issues": issues,
                "suggestions": suggestions,
                "quality_score": self._calculate_content_quality_score(content)
            }
            
        except Exception as e:
            print(f"Error analyzing content quality: {e}")
            return {"error": "Could not analyze content quality"}
    
    def _calculate_content_quality_score(self, content: str) -> int:
        """Calculate overall content quality score (0-100)"""
        try:
            score = 50  # Base score
            
            words = content.split()
            sentences = content.split('.')
            
            # Length score
            if 200 <= len(words) <= 800:
                score += 20
            elif len(words) < 200:
                score -= 20
            else:
                score -= 10
            
            # Action verbs score
            action_verbs = ['developed', 'created', 'built', 'designed', 'implemented', 'managed', 'led', 'achieved', 'improved', 'increased']
            action_verb_count = sum(1 for word in words if word.lower() in action_verbs)
            if action_verb_count >= 5:
                score += 15
            elif action_verb_count >= 3:
                score += 10
            
            # Quantified achievements score
            numbers = re.findall(r'\b\d+\b', content)
            if len(numbers) >= 3:
                score += 15
            elif len(numbers) >= 1:
                score += 10
            
            # Skills section score
            if 'skills' in content.lower():
                score += 10
            
            # Experience section score
            if any(word in content.lower() for word in ['experience', 'work', 'employment']):
                score += 10
            
            return max(0, min(100, score))
            
        except Exception as e:
            return 50  # Default score
    
    def _generate_improvement_suggestions(self, content: str, skills: List[str]) -> List[str]:
        """Generate improvement suggestions for resume"""
        suggestions = []
        
        try:
            # Skill-based suggestions
            if len(skills) < 5:
                suggestions.append("Add more technical skills to strengthen your profile")
            
            # Check for missing common sections
            content_lower = content.lower()
            if 'education' not in content_lower:
                suggestions.append("Add an education section")
            
            if 'projects' not in content_lower:
                suggestions.append("Include relevant projects to showcase your work")
            
            if 'certifications' not in content_lower:
                suggestions.append("Add certifications to demonstrate continuous learning")
            
            # Industry-specific suggestions
            if any(skill.lower() in ['python', 'java', 'javascript'] for skill in skills):
                if 'github' not in content_lower:
                    suggestions.append("Include GitHub profile to showcase your code")
            
            if any(skill.lower() in ['machine learning', 'data science', 'ai'] for skill in skills):
                suggestions.append("Consider adding specific ML projects or datasets you've worked with")
            
            # Format suggestions
            if len(content.split('\n')) < 10:
                suggestions.append("Improve formatting with better section breaks and bullet points")
            
            return suggestions[:5]  # Limit to top 5 suggestions
            
        except Exception as e:
            print(f"Error generating suggestions: {e}")
            return ["Review and improve your resume content"]
    
    def _generate_market_insights(self, skills: List[str]) -> List[Dict[str, str]]:
        """Generate job market insights based on skills"""
        insights = []
        
        try:
            # Skill demand mapping (simplified)
            high_demand_skills = {
                'python': 'Python developers are in high demand across all industries',
                'javascript': 'JavaScript skills are essential for web development roles',
                'react': 'React is one of the most popular frontend frameworks',
                'aws': 'Cloud skills, especially AWS, are highly sought after',
                'machine learning': 'ML and AI skills are rapidly growing in demand',
                'docker': 'Containerization skills are becoming standard in DevOps',
                'kubernetes': 'Kubernetes expertise commands premium salaries',
                'typescript': 'TypeScript is increasingly preferred over JavaScript',
                'node.js': 'Full-stack developers with Node.js are in high demand',
                'sql': 'Database skills remain fundamental across all tech roles'
            }
            
            for skill in skills:
                skill_lower = skill.lower()
                if skill_lower in high_demand_skills:
                    insights.append({
                        'skill': skill,
                        'insight': high_demand_skills[skill_lower],
                        'demand_level': 'high'
                    })
                else:
                    insights.append({
                        'skill': skill,
                        'insight': f'{skill} is a valuable skill in the current market',
                        'demand_level': 'medium'
                    })
            
            return insights[:10]  # Limit to top 10 insights
            
        except Exception as e:
            print(f"Error generating market insights: {e}")
            return []
    
    def _calculate_skill_demand(self, skills: List[str]) -> str:
        """Calculate overall skill demand level"""
        try:
            high_demand_count = 0
            total_skills = len(skills)
            
            high_demand_skills = ['python', 'javascript', 'react', 'aws', 'machine learning', 'docker', 'kubernetes', 'typescript', 'node.js', 'sql']
            
            for skill in skills:
                if skill.lower() in high_demand_skills:
                    high_demand_count += 1
            
            if total_skills == 0:
                return "unknown"
            
            high_demand_ratio = high_demand_count / total_skills
            
            if high_demand_ratio >= 0.6:
                return "high"
            elif high_demand_ratio >= 0.3:
                return "medium"
            else:
                return "low"
                
        except Exception as e:
            return "unknown"
    
    def _recommend_additional_skills(self, current_skills: List[str]) -> List[str]:
        """Recommend additional skills based on current skills"""
        recommendations = []
        
        try:
            current_skills_lower = [skill.lower() for skill in current_skills]
            
            # Skill recommendations based on current skills
            skill_recommendations = {
                'python': ['django', 'flask', 'fastapi', 'pandas', 'numpy'],
                'javascript': ['typescript', 'react', 'vue', 'angular', 'node.js'],
                'react': ['redux', 'next.js', 'typescript', 'jest'],
                'java': ['spring', 'hibernate', 'maven', 'gradle'],
                'aws': ['docker', 'kubernetes', 'terraform', 'ansible'],
                'machine learning': ['tensorflow', 'pytorch', 'scikit-learn', 'pandas'],
                'sql': ['postgresql', 'mongodb', 'redis', 'elasticsearch'],
                'docker': ['kubernetes', 'jenkins', 'gitlab ci', 'terraform']
            }
            
            for skill in current_skills_lower:
                if skill in skill_recommendations:
                    for rec_skill in skill_recommendations[skill]:
                        if rec_skill not in current_skills_lower:
                            recommendations.append(rec_skill.title())
            
            # Remove duplicates and limit
            recommendations = list(set(recommendations))[:5]
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating skill recommendations: {e}")
            return []
    
    def process_resume(self, file_path: str, filename: str, user_role: str = "user") -> Dict[str, any]:
        """Process a resume file and extract information"""
        try:
            # Extract text
            text = self.extract_text(file_path, filename)
            
            if not text.strip():
                raise Exception("No text content found in the file")
            
            # Extract information
            skills = self.extract_skills(text)
            experience = self.extract_experience(text)
            
            # Redact PII for non-recruiter users
            display_text = text
            if user_role != "recruiter":
                display_text = self.redact_pii(text)
            
            # Create embedding
            embedding = self.create_embedding(text)
            
            return {
                "content": text,
                "display_content": display_text,
                "skills": skills,
                "experience": experience,
                "embedding": embedding,
                "word_count": len(text.split()),
                "is_pii_redacted": user_role != "recruiter"
            }
            
        except Exception as e:
            raise Exception(f"Error processing resume: {str(e)}")
    
    def match_job_to_resumes(self, job_embedding: List[float], job_requirements: List[str], 
                           resume_data: List[Dict], top_n: int = 10) -> List[Dict]:
        """Dynamic matching algorithm based on actual content analysis"""
        results = []
        
        for resume in resume_data:
            # Analyze actual resume content dynamically
            resume_content = resume['content']
            analysis_result = self._analyze_resume_vs_job(resume_content, job_requirements)
            
            results.append({
                'resume_id': resume['id'],
                'similarity_score': analysis_result['similarity_score'],
                'matching_skills': analysis_result['matching_skills'],
                'missing_requirements': analysis_result['missing_requirements'],
                'evidence_snippets': analysis_result['evidence_snippets'],
                'candidate_name': self._extract_name_from_resume(resume_content),
                'match_percentage': analysis_result['match_percentage'],
                'experience_level': analysis_result['experience_level'],
                'key_strengths': analysis_result['key_strengths'],
                'skill_gaps': analysis_result['skill_gaps']
            })
        
        # Sort by match percentage (most accurate metric)
        results.sort(key=lambda x: x['match_percentage'], reverse=True)
        return results[:top_n]
    
    def _analyze_resume_vs_job(self, resume_content: str, job_requirements: List[str]) -> Dict:
        """Comprehensive analysis of resume against job requirements"""
        resume_lower = resume_content.lower()
        
        # 1. EXTRACT ACTUAL SKILLS FROM RESUME
        actual_skills = self._extract_actual_skills(resume_content)
        
        # 2. EXTRACT ACTUAL EXPERIENCE FROM RESUME
        experience_info = self._extract_experience_info(resume_content)
        
        # 3. ANALYZE EACH JOB REQUIREMENT
        matching_skills = []
        missing_requirements = []
        evidence_snippets = []
        skill_gaps = []
        
        for req in job_requirements:
            match_result = self._analyze_requirement_match(resume_content, req, actual_skills)
            
            if match_result['matched']:
                matching_skills.append(req)
                if match_result['evidence']:
                    evidence_snippets.append(match_result['evidence'])
            else:
                missing_requirements.append(req)
                skill_gaps.append(match_result['gap_reason'])
        
        # 4. CALCULATE REALISTIC MATCH PERCENTAGE
        match_percentage = self._calculate_realistic_match(
            matching_skills, missing_requirements, job_requirements, 
            experience_info, actual_skills
        )
        
        # 5. IDENTIFY KEY STRENGTHS
        key_strengths = self._identify_key_strengths(resume_content, actual_skills)
        
        return {
            'similarity_score': len(matching_skills) / len(job_requirements) if job_requirements else 0,
            'matching_skills': matching_skills,
            'missing_requirements': missing_requirements,
            'evidence_snippets': evidence_snippets[:3],
            'match_percentage': match_percentage,
            'experience_level': experience_info['level'],
            'key_strengths': key_strengths,
            'skill_gaps': skill_gaps[:3]
        }
    
    def _extract_actual_skills(self, resume_content: str) -> Dict[str, List[str]]:
        """Extract actual skills mentioned in resume with context"""
        resume_lower = resume_content.lower()
        
        # Define skill categories and their keywords
        skill_categories = {
            'programming_languages': [
                'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust', 
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css', 'shell', 'bash'
            ],
            'frameworks_libraries': [
                'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'laravel',
                'bootstrap', 'jquery', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn'
            ],
            'databases': [
                'mysql', 'postgresql', 'mongodb', 'redis', 'oracle', 'sqlite', 'cassandra', 'elasticsearch'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'linux', 'terraform', 'ansible'
            ],
            'tools_software': [
                'figma', 'photoshop', 'sketch', 'jira', 'confluence', 'slack', 'excel', 'powerpoint', 'tableau'
            ],
            'soft_skills': [
                'leadership', 'management', 'communication', 'teamwork', 'problem solving', 'analytical',
                'project management', 'agile', 'scrum'
            ]
        }
        
        found_skills = {category: [] for category in skill_categories}
        
        # Look for skills with context
        sentences = resume_content.split('.')
        for sentence in sentences:
            sentence_lower = sentence.lower().strip()
            if len(sentence_lower) > 10:  # Ignore very short sentences
                for category, skills in skill_categories.items():
                    for skill in skills:
                        if skill in sentence_lower and skill not in found_skills[category]:
                            # Verify it's not part of a larger word
                            if self._is_standalone_skill(sentence_lower, skill):
                                found_skills[category].append(skill)
        
        return found_skills
    
    def _is_standalone_skill(self, text: str, skill: str) -> bool:
        """Check if skill appears as standalone word/phrase"""
        import re
        # Create word boundary pattern for the skill
        pattern = r'\b' + re.escape(skill.replace('.', '\\.')).replace('\\ ', '\\s+') + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _extract_experience_info(self, resume_content: str) -> Dict:
        """Extract actual experience information from resume"""
        import re
        resume_lower = resume_content.lower()
        
        # Look for experience indicators
        experience_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience\s+(?:of\s+)?(\d+)\+?\s*years?',
            r'(\d+)\s*years?\s+(?:of\s+)?(?:professional\s+)?experience',
            r'working\s+(?:for\s+)?(\d+)\+?\s*years?'
        ]
        
        years_found = []
        for pattern in experience_patterns:
            matches = re.findall(pattern, resume_lower)
            years_found.extend([int(match) for match in matches if match.isdigit()])
        
        # Determine experience level
        if 'fresher' in resume_lower or 'fresh graduate' in resume_lower or 'no experience' in resume_lower:
            level = 'fresher'
            years = 0
        elif years_found:
            years = max(years_found)  # Take the highest mentioned years
            if years <= 1:
                level = 'entry'
            elif years <= 3:
                level = 'junior'
            elif years <= 7:
                level = 'mid'
            else:
                level = 'senior'
        else:
            # Try to infer from job titles
            if any(word in resume_lower for word in ['senior', 'lead', 'manager', 'director', 'head']):
                level = 'senior'
                years = 5  # Estimate
            elif any(word in resume_lower for word in ['junior', 'associate', 'assistant']):
                level = 'junior'
                years = 2  # Estimate
            else:
                level = 'unknown'
                years = 1  # Default estimate
        
        return {
            'level': level,
            'years': years,
            'years_mentioned': years_found
        }
    
    def _analyze_requirement_match(self, resume_content: str, requirement: str, actual_skills: Dict) -> Dict:
        """Analyze if a specific job requirement is met"""
        import re
        
        req_lower = requirement.lower()
        resume_lower = resume_content.lower()
        
        # Extract key terms from requirement
        # Remove common words and focus on technical terms
        stop_words = {'with', 'and', 'or', 'in', 'of', 'the', 'a', 'an', 'to', 'for', 'experience', 'knowledge', 'skills'}
        req_words = [word.strip('.,()+-') for word in req_lower.split() if word not in stop_words and len(word) > 2]
        
        # Look for matches in actual skills first
        found_matches = []
        all_skills = [skill for skills_list in actual_skills.values() for skill in skills_list]
        
        for word in req_words:
            # Direct skill match
            if any(word in skill.lower() for skill in all_skills):
                found_matches.append(word)
            # Text content match
            elif word in resume_lower:
                found_matches.append(word)
        
        # Calculate match confidence
        if len(req_words) == 0:
            match_confidence = 0
        else:
            match_confidence = len(found_matches) / len(req_words)
        
        # Find evidence in resume
        evidence = None
        if found_matches:
            sentences = resume_content.split('.')
            for sentence in sentences:
                sentence_clean = sentence.strip()
                if len(sentence_clean) > 20:
                    # Check if sentence contains multiple matched terms
                    matches_in_sentence = sum(1 for match in found_matches if match in sentence.lower())
                    if matches_in_sentence >= len(found_matches) * 0.5:  # At least 50% of matches
                        evidence = sentence_clean + '.'
                        break
        
        # Determine if requirement is met (threshold: 60%)
        is_matched = match_confidence >= 0.6
        
        gap_reason = None
        if not is_matched:
            missing_terms = [word for word in req_words if word not in [match for match in found_matches]]
            gap_reason = f"Missing: {', '.join(missing_terms[:3])}"  # Show top 3 missing terms
        
        return {
            'matched': is_matched,
            'confidence': match_confidence,
            'evidence': evidence,
            'gap_reason': gap_reason,
            'found_terms': found_matches
        }
    
    def _calculate_realistic_match(self, matching_skills: List[str], missing_requirements: List[str], 
                                 all_requirements: List[str], experience_info: Dict, actual_skills: Dict) -> int:
        """Calculate realistic match percentage based on comprehensive analysis"""
        
        if not all_requirements:
            return 0
        
        # Base skill match ratio
        skill_match_ratio = len(matching_skills) / len(all_requirements)
        
        # Experience level adjustment
        experience_adjustment = 0
        exp_level = experience_info['level']
        
        # Check if job requires specific experience level
        requirements_text = ' '.join(all_requirements).lower()
        
        if 'senior' in requirements_text or '5+' in requirements_text or '7+' in requirements_text:
            required_level = 'senior'
        elif '3+' in requirements_text or '4+' in requirements_text:
            required_level = 'mid'
        elif '1+' in requirements_text or '2+' in requirements_text:
            required_level = 'junior'
        else:
            required_level = 'any'
        
        # Experience level matching
        if required_level != 'any':
            if exp_level == 'fresher' and required_level in ['senior', 'mid']:
                experience_adjustment = -0.3  # 30% penalty
            elif exp_level == 'junior' and required_level == 'senior':
                experience_adjustment = -0.15  # 15% penalty
            elif exp_level == 'senior' and required_level in ['junior', 'entry']:
                experience_adjustment = 0.1  # 10% bonus (overqualified)
        
        # Skill diversity bonus
        total_skills_found = sum(len(skills) for skills in actual_skills.values())
        diversity_bonus = min(0.1, total_skills_found / 20)  # Up to 10% bonus for diverse skills
        
        # Calculate final percentage
        final_score = skill_match_ratio + experience_adjustment + diversity_bonus
        final_percentage = int(max(5, min(95, final_score * 100)))  # 5-95% range
        
        return final_percentage
    
    def _identify_key_strengths(self, resume_content: str, actual_skills: Dict) -> List[str]:
        """Identify top strengths from resume"""
        strengths = []
        
        # Most mentioned skills
        all_skills = []
        for category, skills in actual_skills.items():
            if skills:  # If category has skills
                all_skills.extend(skills[:2])  # Top 2 from each category
        
        # Add top skills as strengths
        strengths.extend(all_skills[:5])
        
        # Look for achievements or standout phrases
        resume_lower = resume_content.lower()
        achievement_indicators = [
            'achieved', 'improved', 'increased', 'reduced', 'led', 'managed', 
            'developed', 'created', 'built', 'designed', 'implemented'
        ]
        
        sentences = resume_content.split('.')
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) > 30 and any(indicator in sentence.lower() for indicator in achievement_indicators):
                strengths.append(sentence_clean[:80] + '...' if len(sentence_clean) > 80 else sentence_clean)
                if len(strengths) >= 8:  # Limit total strengths
                    break
        
        return strengths[:5]  # Return top 5 strengths
    
    def _extract_name_from_resume(self, text: str) -> str:
        """Simple name extraction from resume"""
        lines = text.split('\n')[:5]  # Check first 5 lines
        
        for line in lines:
            line = line.strip()
            # Look for lines that might be names (capitalized words, reasonable length)
            words = line.split()
            if (len(words) >= 2 and len(words) <= 4 and 
                all(word.replace('.', '').isalpha() and word[0].isupper() for word in words if word)):
                if len(line) < 50:  # Names shouldn't be too long
                    return line
        
        return "Unknown Candidate"
