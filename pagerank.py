import os
import random
import re
import sys

from pomegranate import *

DAMPING = 0.85
SAMPLES = 10000
CONVERGE_VALUE = 0.001


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


def normalize_distribution(distribution):

    # Use the sum of distribution values to determine alpha (normalising value)
    totalProbability = 0
    for value in distribution.values():
        totalProbability += value

    # alpha = 1/totalProbability
    for value in distribution.values():
        value = value/totalProbability

    return distribution


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    probabilityDistribution = dict()

    # Damping factor probablity
    for nextPage in corpus[page]:
        probabilityDistribution[nextPage] = 1/len(corpus[page])

    # Calculate the random landing probability
    # This is added to every page in the corpus
    notDProbability = (1-damping_factor)/len(corpus)

    for keyPage in corpus:
        if not keyPage in probabilityDistribution:
            probabilityDistribution[keyPage] = 0

        probabilityDistribution[keyPage] += notDProbability

    # Normalising the distribution
    probabilityDistribution = normalize_distribution(probabilityDistribution)

    return probabilityDistribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRanks = dict()
    model = MarkovChain()

    # The first page that is randomly selected
    firstOrder = dict()
    countTotalPages = len(corpus)

    for page in corpus.keys():
        firstOrder[page] = 1/countTotalPages

    d1 = DiscreteDistribution(
        firstOrder
    )

    # The next page a random surfer selects after a page
    secondOrder = []

    for pageKey, pageSet in corpus:
        transitionModel = transition_model(corpus, pageKey, damping_factor)
        for nextPage in pageSet:
            secondOrderList = [pageKey, nextPage, transitionModel[nextPage]]
            secondOrder.append(secondOrderList)

    d2 = ConditionalProbabilityTable(
        secondOrder
    )

    # Create a Markov Chain
    model = MarkovChain([d1, d2])

    # Using model to create samples

    return pageRanks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRanks = dict()
    totalCountPages = len(corpus)
    for page in corpus:
        pageRanks[page] = 1/totalCountPages

    # Apply the iterative formula
    converge = False
    while not converge:
        copyPageRanks = copy
        tempSum = dict()
        for pageKey, pageNextPages in corpus:
            # Add the pageNextPage
            countPagesI = len(pageNextPages)
            for nextPage in pageNextPages:
                if not nextPage in tempSum:
                    tempSum[nextPage] = 0
                tempSum[nextPage] += pageRanks[pageKey]/countPagesI

        for pageKey, sumDamping in tempSum:
            tempPageRank = ((1-damping_factor) /
                            totalCountPages) + damping_factor*sumDamping
            if (tempPageRank - pageRanks[pageKey]) < CONVERGE_VALUE:
                converge = True
                break
            else:
                pageRanks[pageKey] = tempPageRank

    return pageRanks


if __name__ == "__main__":
    main()
