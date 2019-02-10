# Deep Sloop READme

## Main Workflow and Scripts

The data used in this project in order to create the DeepSloop model was originally sourced from bpRNA, a meta-database desribed in the following paper: https://www.ncbi.nlm.nih.gov/pubmed/29746666. 

The raw data that was retrieved from the database consisted of two separate files:
1. allSegments_Raw.txt which contains tab separated data for RNA stem loop segments that are the                       heavily bonded portions of the hairpin structure. 
2. allHairpins_Raw.txt which contains tab separated data for RNA hairpin loops.

The process of consolidating the information contained in these two files in order to produce a FASTA file full of RNA stem loops is automated by the Generate_Sloop_Dataset.py script. This script is responsible for piecing together RNA molecules using RNA ID values,  filtering the hairpins for bases outside of the traditional AUCG, and enforcing user-defined requirements for loop and segment length.

Optionally, a user may choose to subject their resulting FASTA file to another round of filtering, this time enforcing a fraction of left parenthesis on the left half of the molecule on the dot-bracket structure of the molecule produced by the RNAfold software.

Once the FASTA file of stem loops (sloops) has been obtained, Initialize_Data_Infrastrcuture.py may be used to initialize a data infrastructure containing testing, training, and validation data. 

Beyond this, it is time to train up and apply the DeepSloop model.
DeepSloop_Training.py : Script that can be used to produce the DeepSloop model with optimal 		                              hyperparameters
DeepSloop_Utils.py : Contains utility files for the project for purposes of filetype conversion, data    				processing, etc.
DeepSloop_HyperOpt.py : Script with the DeepSloop model configured for hyperparameter tuning                     			        with the HyperOpt package

As a proof of concept for the model’s potential applications, the script HairpinFinder_D1.py integrates the DeepSloop model into an algorithm that can scan variable length RNA sequences in order to assign a per-base probability score for the prescence of an RNA hairpin. This may be applied to the HOTAIR_Domain1.fasta file contained in the Data directory.

### For more information about the scripts contained in this project

Every file in this project has full documentation for its subroutines. This is a complete public check-in with working versions of all project files.
