import numpy as np
import math

def clr(x, eps=1e-12):
    """
    Computes the centered log-ratio (CLR) transformation
    """
    x = np.array(x) + eps
    gm = np.exp(np.mean(np.log(x)))
    return np.log(x / gm)

def aitchison_distance(p, q, eps=1e-12):
    """
    Computes the Aitchison distance between two probability distributions
    """
    clr_p = clr(p, eps)
    clr_q = clr(q, eps)
    return np.linalg.norm(clr_p - clr_q)

def aitchison_similarity(p, q):
    """
    Computes an Aitchison-based similarity measure
    Uses an exponential decay so that a distance of 0 gives similarity 1
    """
    d = aitchison_distance(p, q)
    return math.exp(-d)


def compute_club_topic_distributions(data_distr, dictionary, lda_model, num_topics):
    """
    Computes and averages the topic distributions per club given the data and an LDA model
    Returns a dictionary mapping club names to their average topic distribution
    """
    club_topic_vectors = {}
    for key, doc in data_distr.items():
        _, club = key.split('_', 1)
        bow = dictionary.doc2bow(doc)
        # Get topic distribution (ensure all topics are present)
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
        topic_vec = np.zeros(num_topics)
        for topic_id, prob in doc_topics:
            topic_vec[topic_id] = prob
        club_topic_vectors.setdefault(club, []).append(topic_vec)
    
    # Average topic distributions for each club
    club_avg_topic_distributions = {club: np.mean(vecs, axis=0)
                                    for club, vecs in club_topic_vectors.items()}
    return club_avg_topic_distributions