# Methodology

Each isolate is represented by one graph of Bakta CDS features. Edges connect neighboring genes within a contig. GenoContext-GNN uses two GraphSAGE layers; DeepSets, logistic regression, and random forest provide baselines using the same training isolates.

The K. pneumoniae experiment reads exact per-antibiotic train, validation, and test ID files for seeds 0–9. Training isolates alone fit feature vocabularies and model weights. Validation isolates select neural checkpoints by resistant-class F1 and select a binary threshold that maximizes resistant-class F1 for every model. Test isolates are used only for final evaluation. F1, AUROC, AUPRC, accuracy, balanced accuracy, precision, recall, specificity, MCC, and log loss are reported for validation and test partitions.

Raw and AMR-term-masked annotation features, plus real and shuffled within-contig graph edges, are separate study modes. An existing GFF3 skips Bakta. An invalid existing GFF3 raises an error; Bakta runs only if the file is missing. Feature IDs and locus tags provide traceability, not predictive values.
