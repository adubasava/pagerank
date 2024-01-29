import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])     
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = dict()
    
    # If no outgoing links
    if len(corpus[page]) == 0:
        prob = 1 / len(corpus.keys())
        for key in corpus:
            probability_distribution[key] = prob
    # If there are outgoing links
    else:
        prob1 = damping_factor / len(corpus[page]) 
        prob2 = (1 - damping_factor) / len(corpus.keys())
        prob = prob1 + prob2
        for key in corpus:
            probability_distribution[key] = prob2
        for x in corpus[page]:            
            probability_distribution[x] = prob
    
    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # First random page
    sample = random.choice(list(corpus.keys()))
    pr_dict = dict()
    pr_dict[sample] = 1

    # Main loop 
    for i in range(1, n):
        new_sample = transition_model(corpus, sample, damping_factor)
        page_next = random.choices(list(new_sample.keys()), weights=list(new_sample.values()), k=1)[0]
        if page_next in list(pr_dict.keys()):
            pr_dict[page_next] = pr_dict[page_next] + 1
        else:
            pr_dict[page_next] = 1
        sample = page_next
    
    for key in pr_dict:
        pr_dict[key] = pr_dict[key] / n
    
    return pr_dict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    iterate_dict = dict()

    # First distribution
    for key in corpus:
        iterate_dict[key] = 1 / len(corpus.keys())

    while True:
        count = 0
        for key in corpus:            
            sum = 0
            for page in corpus:
                if key in corpus[page]:
                    sum = sum + iterate_dict[page] / len(corpus[page])
            new = (1 - damping_factor) / len(corpus.keys()) + damping_factor * sum
            if abs(iterate_dict[key] - new) < 0.0001:
                count += 1
            iterate_dict[key] = new 
        if count == len(corpus.keys()):
            break
    
    # Normalization
    total = 0
    for key in iterate_dict:
        total += iterate_dict[key]
    for key in iterate_dict:
        iterate_dict[key] = iterate_dict[key] / total
    
    return iterate_dict
   

if __name__ == "__main__":
    main()
