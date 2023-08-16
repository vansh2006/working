import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

class Family:
    def __init__(self, child, mother, father):
        self.child = child
        self.mother = mother
        self.father = father

class Person:
    def __init__(self, name, trait):
        self.name = name
        self.trait = trait


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Probability of the combination of these events being true
    p = 1

    for person in list(people.values()):

        trait = person["name"] in have_trait

        gene = 0
        if person["name"] in one_gene:
            gene = 1
        elif person["name"] in two_genes:
            gene = 2

        parents = find_parents(people, person)
        mother = parents[0]
        father = parents[1]

        if mother and father:
            # Find probability that person inherits one copy of the gene from their parents
            mother_and_father_chances = probability_to_inherit(mother, father, one_gene, two_genes)
            inherit_from_mother = mother_and_father_chances[0]
            inherit_from_father = mother_and_father_chances[1]
            not_from_mother = 1 - inherit_from_mother
            not_from_father = 1 - inherit_from_father

            # Inherit zero copies means to not get gene from father and not get gene from mother
            p_to_inherit = not_from_mother * not_from_father

            # Person inherited one gene either from mother and not father not mother and far
            # Probability person inherited one gene = (a ^ !b) + (!a ^ b)

            if gene == 1:
                p_to_inherit = (inherit_from_mother * not_from_father) + (inherit_from_father * not_from_mother)
            
            # Inherit two copies means to inherit one from mother and father

            elif gene == 2:
                p_to_inherit = inherit_from_father * inherit_from_mother

            p *= p_to_inherit

            # Probability person had trait given number of genes
            p *= PROBS["trait"][gene][trait]


        else:
            # Probability that `person`` has `gene` genes
            p *= PROBS["gene"][gene]
            # Probability that `person` has trait or not
            p *= PROBS["trait"][gene][trait]
    return p

def probability_to_inherit(mother, father, one_gene, two_genes):
    # Chances of getting one gene given zero, one, or two copies from a parent
    zero_chance = PROBS["mutation"]
    one_chance = 0.5
    two_chance = 1 - PROBS["mutation"]

    mother_chance = zero_chance
    if mother in one_gene:
        mother_chance = one_chance
    elif mother in two_genes:
        mother_chance = two_chance

    father_chance = zero_chance
    if father in one_gene:
        father_chance = one_chance
    elif father in two_genes:
        father_chance = two_chance

    return (mother_chance, father_chance)


def find_parents(people, person):
    mother = None
    father = None
    for individual in list(people.values()):
        if person["mother"] == individual["name"]:
            mother = individual
        if person["father"] == individual["name"]:
            father = individual
        if mother and father:
            return (mother["name"], father["name"])
    return (mother, father)
    

def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """

    # Put a loop above your current loop, same for person in probabilities 
    # But print the person, which gene, prob of that gene, which trait, prob of that trait. And see.

    for person in probabilities:
        if person in one_gene:
            probabilities[person]["gene"][1] += p
        elif person in two_genes:
            probabilities[person]["gene"][2] += p
        else:
            probabilities[person]["gene"][0] += p

        if person in have_trait:
            probabilities[person]["trait"][True] += p
        else:
            probabilities[person]["trait"][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in probabilities:
        sum_gene = 0
        sum_trait = 0

        for thing in probabilities[person]["gene"]:
            sum_gene += probabilities[person]["gene"][thing]
        for stuff in probabilities[person]["trait"]:
            sum_trait += probabilities[person]["trait"][stuff]

        factor_gene = 1 / sum_gene
        factor_trait = 1 / sum_trait

        for num_gene in probabilities[person]["gene"]:
            probabilities[person]["gene"][num_gene] *= factor_gene
        for type_trait in probabilities[person]["trait"]:
            probabilities[person]["trait"][type_trait] *= factor_trait


    




if __name__ == "__main__":
    main()