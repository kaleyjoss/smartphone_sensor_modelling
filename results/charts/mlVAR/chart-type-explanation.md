1. The Contemporaneous Network (Within-Person)
The contemporaneous network represents how variables are associated with each other within the same measurement window (e.g., within the same hour or day) after controlling for temporal effects from the previous time point.

    Edge Meaning: Edges represent partial correlations between the innovations (or residuals)—the "shocks" or new information that enters the system at time t that cannot be explained by the state of the system at time t−1.
    Timescale: This network captures fast-acting dynamics that occur at a timescale quicker than the interval between measurements. For instance, if a person is measured every three hours, any interaction that happens within minutes will appear in this network.
    Structure: It is an undirected graphical model (specifically a Pairwise Markov Random Field).

2. The Between-Person Network (Between-Subjects)
The between-person network represents the stable, long-term associations between individuals' averages across the entire study period.

    Edge Meaning: Edges represent conditional associations between subject-specific means (random intercepts). For example, if people who are generally more "worried" on average also tend to be generally more "depressed" on average across all weeks, an edge will connect these nodes in this network.
    Interpretation: This network is driven by individual differences and is conceptually similar to a cross-sectional network constructed from a single time point in a large sample. It reflects the "trait-like" structure of a group rather than the dynamic "state-like" processes of an individual.
    Methodological Note: Research indicates that the mlVAR package may have biases in recovering this specific network, particularly when the number of measurement points per person is low, as it uses within-person sample means as proxies for population means.


3. Temporal Network
The temporal network represents how one variable influences the next timepoint of that variable. 
    Edge meaning: The self-loop edges represent how strong the correlation is between a variable yesterday and that same variable today. Cross-variable edges represent how strong a correlation is between a variable yesterday and a different variable today.