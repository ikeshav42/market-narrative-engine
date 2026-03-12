"""News relevance scoring using GloVe word embeddings."""

import re
import math
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class RelevanceResult:
    overall_score: float
    confidence: float
    breakdown: dict = field(default_factory=dict)
    is_breaking: bool = False


class GloveEmbeddings:
    """Loads GloVe word vectors once."""

    _instance = None
    _embeddings: dict = None
    _embedding_dim: int = 50

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._embeddings = {}
            cls._instance._embedding_dim = 50
        return cls._instance

    def load(self, filepath: str, embedding_dim: int = 50) -> bool:
        if self._embeddings:
            return True

        self._embedding_dim = embedding_dim

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.rstrip().split(' ')
                    word = parts[0]
                    vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    self._embeddings[word] = vector
            return True
        except Exception:
            return False

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        return self._embeddings.get(word.lower())

    def __contains__(self, word: str) -> bool:
        return word.lower() in self._embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def is_loaded(self) -> bool:
        return len(self._embeddings) > 0


_glove = GloveEmbeddings()


class RelevanceScorer:
    """Scores news articles for relevance to a company using GloVe embeddings."""

    WEIGHTS = {
        'direct_match': 0.45,
        'semantic': 0.40,
        'source': 0.10,
        'temporal': 0.05
    }

    SOURCE_PRIORS = {
        'reuters': 0.95,
        'bloomberg': 0.95,
        'financial times': 0.92,
        'wall street journal': 0.92,
        'wsj': 0.92,
        'cnbc': 0.90,
        'barrons': 0.88,
        'marketwatch': 0.85,
        'seeking alpha': 0.82,
        'yahoo finance': 0.80,
        'yahoo': 0.80,
        'benzinga': 0.75,
        'investorplace': 0.72,
        'motley fool': 0.70,
        'fool': 0.70,
        'zacks': 0.70,
        'tipranks': 0.70,
        'thestreet': 0.68,
        'investor business daily': 0.75,
        'ibd': 0.75,
    }

    BREAKING_KEYWORDS = [
        'breaking', 'just in', 'alert', 'urgent', 'developing',
        'announces', 'reports', 'confirms', 'surges', 'plunges',
        'soars', 'crashes', 'jumps', 'drops', 'beats', 'misses'
    ]

    TEMPORAL_HALF_LIFE_HOURS = 6.0

    def __init__(self, glove_path: str = 'data/glove.6B.50d.txt'):
        self._glove = _glove
        self._glove_loaded = self._glove.load(glove_path)
        self._company_vector_cache: dict = {}

    def _preprocess_text(self, text: str) -> str:
        if not text:
            return ''
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Mean pooling of word embeddings."""
        words = self._preprocess_text(text).split()
        vectors = []

        for word in words:
            vec = self._glove.get_vector(word)
            if vec is not None:
                vectors.append(vec)

        if not vectors:
            return np.zeros(self._glove.embedding_dim, dtype=np.float32)

        return np.mean(np.stack(vectors), axis=0)

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """cos(theta) = (A · B) / (||A|| × ||B||)"""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _compute_direct_match_score(self, text: str, company_info: dict) -> float:
        if not text or not company_info:
            return 0.0

        text_lower = text.lower()
        score = 0.0

        ticker = company_info.get('ticker', '').upper()
        if ticker:
            ticker_pattern = r'\b' + re.escape(ticker.lower()) + r'\b'
            ticker_matches = len(re.findall(ticker_pattern, text_lower))
            if ticker_matches > 0:
                score += 0.5
                score += min(0.1, ticker_matches * 0.02)

        company_name = company_info.get('shortName', '')
        if company_name:
            name_words = self._preprocess_text(company_name).split()
            significant_words = [w for w in name_words if len(w) > 2 and w not in
                               {'inc', 'corp', 'ltd', 'llc', 'the', 'and', 'company', 'co'}]

            if significant_words:
                matches = sum(1 for word in significant_words
                            if re.search(r'\b' + re.escape(word) + r'\b', text_lower))
                if matches > 0:
                    score += (matches / len(significant_words)) * 0.3

        return min(1.0, score)

    def _compute_semantic_score(self, text: str, company_info: dict) -> float:
        if not self._glove_loaded:
            return 0.5

        ticker = company_info.get('ticker', '')

        if ticker and ticker in self._company_vector_cache:
            company_vec = self._company_vector_cache[ticker]
        else:
            company_context = ' '.join([
                company_info.get('shortName', ''),
                company_info.get('sector', ''),
                company_info.get('industry', ''),
                company_info.get('longBusinessSummary', '')
            ])
            company_vec = self._text_to_vector(company_context)
            if ticker:
                self._company_vector_cache[ticker] = company_vec

        article_vec = self._text_to_vector(text)
        similarity = self._cosine_similarity(company_vec, article_vec)

        # Map [-1, 1] to [0, 1]
        return (similarity + 1) / 2

    def _get_source_prior(self, source: str) -> float:
        if not source:
            return 0.5

        source_lower = source.lower().strip()

        if source_lower in self.SOURCE_PRIORS:
            return self.SOURCE_PRIORS[source_lower]

        for known_source, prior in self.SOURCE_PRIORS.items():
            if known_source in source_lower or source_lower in known_source:
                return prior

        return 0.50

    def _compute_temporal_score(self, publish_time: Optional[str]) -> float:
        if not publish_time:
            return 0.5

        try:
            if isinstance(publish_time, datetime):
                pub_dt = publish_time
            else:
                for fmt in ['%Y-%m-%d %H:%M', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d']:
                    try:
                        pub_dt = datetime.strptime(publish_time, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return 0.5

            now = datetime.now(timezone.utc)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            else:
                pub_dt = pub_dt.astimezone(timezone.utc)

            hours_elapsed = (now - pub_dt).total_seconds() / 3600

            if hours_elapsed < 0:
                return 1.0

            decay = math.pow(0.5, hours_elapsed / self.TEMPORAL_HALF_LIFE_HOURS)
            return max(0.0, min(1.0, decay))
        except Exception:
            return 0.5

    def _is_breaking_news(self, title: str, summary: str = '') -> bool:
        text = (title + ' ' + summary).lower()
        return any(kw in text for kw in self.BREAKING_KEYWORDS)

    def _compute_confidence(self, breakdown: dict, text_length: int) -> float:
        scores = list(breakdown.values())

        text_confidence = min(1.0, text_length / 500) * 0.3

        if scores:
            mean_score = sum(scores) / len(scores)
            variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
            agreement_confidence = max(0.0, 1.0 - variance * 2) * 0.4
        else:
            agreement_confidence = 0.0

        direct_score = breakdown.get('direct_match', 0.5)
        signal_confidence = abs(direct_score - 0.5) * 2 * 0.3

        return min(1.0, text_confidence + agreement_confidence + signal_confidence)

    def score(self, article: dict, company_info: dict) -> RelevanceResult:
        title = article.get('title', '')
        summary = article.get('summary', '')
        source = article.get('source', '')
        pub_time = article.get('datetime', '')

        full_text = f"{title} {summary}"

        breakdown = {
            'direct_match': self._compute_direct_match_score(full_text, company_info),
            'semantic': self._compute_semantic_score(full_text, company_info),
            'source': self._get_source_prior(source),
            'temporal': self._compute_temporal_score(pub_time)
        }

        overall_score = sum(
            breakdown[component] * weight
            for component, weight in self.WEIGHTS.items()
        )

        is_breaking = self._is_breaking_news(title, summary)

        # Boost breaking news by 20%
        if is_breaking:
            overall_score = min(1.0, overall_score + 0.2)

        confidence = self._compute_confidence(breakdown, len(full_text))

        return RelevanceResult(
            overall_score=round(overall_score, 3),
            confidence=round(confidence, 3),
            breakdown={k: round(v, 3) for k, v in breakdown.items()},
            is_breaking=is_breaking
        )

    def score_batch(self, articles: list, company_info: dict) -> list:
        return [self.score(article, company_info) for article in articles]

    def rank_articles(self, articles: list, company_info: dict) -> list:
        scored = [(article, self.score(article, company_info)) for article in articles]
        return sorted(scored, key=lambda x: x[1].overall_score, reverse=True)


_scorer_instance = None


def get_scorer(glove_path: str = 'data/glove.6B.50d.txt') -> RelevanceScorer:
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = RelevanceScorer(glove_path=glove_path)
    return _scorer_instance
