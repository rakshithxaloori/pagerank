import os
import random
import re
import sys

from collections import Counter
from random import choice, choices

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
    for key in distribution.keys():
        distribution[key] = distribution[key]/totalProbability

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
    
    transitionModelListDict = dict()
    allPagesList = list()
    data = list()

    for page in corpus.keys():
        allPagesList.append(page)

    # Store transition models in a dictionary, transitionModelListDict[page] = (pagesList, pageWeightsList)
    for page in corpus.keys():
        pageTransitionModel = transition_model(corpus, page, damping_factor)
        pages = []
        pageWeights = []
        for nextPage, weight in pageTransitionModel.items():
            pages.append(nextPage)
            pageWeights.append(weight)
        
        transitionModelListDict[page] = (pages, pageWeights)

    # First ramdomly choose a page
    prevPage = choice(allPagesList)
    data.append(prevPage)

    # Choose the next values from transitionModelListDict
    for i in range(SAMPLES):
        nextPage = choices(population = transitionModelListDict[prevPage][0], weights = transitionModelListDict[prevPage][1], k = 1)[0]
        prevPage = nextPage
        data.append(prevPage)


    # Count the samples from data
    countDict = Counter(data)
    for page in corpus.keys():
        pageRanks[page] = countDict[page]/SAMPLES

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

    # Dictionary to track which pageRanks have converged
    converge = dict()
    for page in corpus.keys():
        converge[page] = False

    # Boolean to check if all pageRanks have converged
    totalConverge = False

    # Keep applying the iterative formula until convergence
    while not totalConverge:

        sigmaLink = dict()
        # sigmaLink stores the sigma part in the iterative formula
        for pageKey, linkedPages in corpus.items():
            countPagesI = len(linkedPages)
            for nextPage in linkedPages:
                if not nextPage in sigmaLink.keys():
                    sigmaLink[nextPage] = 0
                sigmaLink[nextPage] += (pageRanks[pageKey]/countPagesI)

        # Applying the iterative formula to each page
        for pageKey in corpus.keys():
            tempPageRank = ((1-damping_factor)/totalCountPages)
            if pageKey in sigmaLink:
                tempPageRank += (damping_factor*sigmaLink[pageKey])

            # Check for convergence
            if abs((tempPageRank - pageRanks[pageKey])) < CONVERGE_VALUE:
                # Convergence attained
                converge[pageKey] = True
            else:
                # Update the value
                pageRanks[pageKey] = tempPageRank

        # Checking if all pageRanks have converged
        breakStatement = False
        for convergeBool in converge.values():
            if convergeBool == False:
                breakStatement = True
                break

        if breakStatement:
            # All values haven't converged yet
            continue
        else:
            # All values have converged
            totalConverge = True
            break

    pageRanks = normalize_distribution(pageRanks)
    return pageRanks


if __name__ == "__main__":
    main()
