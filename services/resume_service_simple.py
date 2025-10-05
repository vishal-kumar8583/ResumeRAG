import os
import json
import re
from typing import List, Dict, Tuple, Optional
import PyPDF2
import docx
import random
import hashlib
from collections import Counter

class ResumeParsingService:
    def __init__(self, model_name: str = "basic"):
        # Use basic approach without heavy ML dependencies for now
        self.model_name = model_name
        
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
            elif pii_type == 'linkedin':
                redacted_text = re.sub(pattern, '[LINKEDIN_REDACTED]', redacted_text)
            elif pii_type == 'github':
                redacted_text = re.sub(pattern, '[GITHUB_REDACTED]', redacted_text)
            elif pii_type == 'portfolio':
                redacted_text = re.sub(pattern, '[PORTFOLIO_REDACTED]', redacted_text)
        
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
        
        # Extract additional skills using pattern matching
        additional_skills = self._extract_skills_with_patterns(text)
        found_skills.extend(additional_skills)
        
        # Remove duplicates and return
        return list(set(found_skills))
    
    def _is_standalone_skill(self, text: str, skill: str) -> bool:
        """Check if skill appears as standalone word/phrase"""
        # Create word boundary pattern for the skill
        pattern = r'\b' + re.escape(skill.replace('.', '\\.')).replace('\\ ', '\\s+') + r'\b'
        return bool(re.search(pattern, text, re.IGNORECASE))
    
    def _extract_skills_with_patterns(self, text: str) -> List[str]:
        """Extract skills using pattern matching"""
        additional_skills = []
        
        # Skills patterns
        skills_patterns = [
            r'skills?[:\s]+([^.]+)',
            r'technologies?[:\s]+([^.]+)',
            r'expertise[:\s]+([^.]+)',
            r'proficient\s+in[:\s]+([^.]+)',
        ]
        
        for pattern in skills_patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                # Split by common separators
                skills_in_match = re.split(r'[,;|•\n]', match)
                for skill in skills_in_match:
                    skill = skill.strip()
                    if len(skill) > 2 and len(skill) < 20:
                        additional_skills.append(skill.title())
        
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
        """Create simple text-based embedding"""
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
            print(f"Error creating embedding: {e}")
            return [0.1] * 50
    
    def similarity_search(self, query_embedding: List[float], resume_embeddings: List[Tuple[int, List[float]]], k: int = 5) -> List[Tuple[int, float]]:
        """Find similar resumes based on cosine similarity"""
        if not resume_embeddings:
            return []
        
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
            
            # Combine scores using weighted average
            combined_results = []
            for resume_id, semantic_score in semantic_scores:
                fuzzy_score = fuzzy_scores.get(resume_id, 0.0)
                
                # Weighted combination: 70% semantic, 30% fuzzy
                combined_score = 0.7 * semantic_score + 0.3 * fuzzy_score
                
                # Find resume data
                resume_info = next((r for r in resume_data if r['id'] == resume_id), None)
                if resume_info:
                    combined_results.append({
                        'resume_id': resume_id,
                        'semantic_score': semantic_score,
                        'fuzzy_score': fuzzy_score,
                        'combined_score': combined_score,
                        'resume_info': resume_info
                    })
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            return combined_results[:k]
            
        except Exception as e:
            print(f"Error in advanced semantic search: {e}")
            return []
    
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
                    filename_score = self._simple_fuzzy_match(query_lower, resume['filename'].lower())
                    scores.append(filename_score)
                
                # Content matching
                content_score = self._simple_fuzzy_match(query_lower, content.lower())
                scores.append(content_score)
                
                # Take the maximum score
                fuzzy_scores[resume_id] = max(scores) if scores else 0.0
                
        except Exception as e:
            print(f"Error calculating fuzzy scores: {e}")
        
        return fuzzy_scores
    
    def _simple_fuzzy_match(self, query: str, text: str) -> float:
        """Simple fuzzy matching based on word overlap"""
        query_words = set(query.split())
        text_words = set(text.split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(text_words)
        union = query_words.union(text_words)
        
        return len(intersection) / len(union) if union else 0.0
    
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
    
    def _extract_experience_info(self, resume_content: str) -> Dict:
        """Extract actual experience information from resume"""
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
    
    def _analyze_requirement_match(self, resume_content: str, requirement: str, actual_skills: Dict) -> Dict:
        """Analyze if a specific job requirement is met"""
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

