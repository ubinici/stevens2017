# Testing Word Learning Models by Stevens et al. (2017)
This is a (Work-In-Progress) Python implementation of PbV and Pursuit models by Stevens et al. (2017). The paper can be found [here](https://onlinelibrary.wiley.com/doi/10.1111/cogs.12416).

The project is still under development. For the time being, I have replicated and tested the PbV and Pursuit models as explained in the paper. Additionally, I have created a more sophisticated version of PbV that stores already existing word-referent pairs to avoid overlaps. Furthermore, I have also implemented a roulette-wheel selection method for the initialization with a new referent upon punishing the absentee pair for the Pursuit algorithm, instead of strictly random initialization.

The models are trained on sentence-referent pairs taken from [Rollins corpus](https://childes.talkbank.org/access/Eng-NA/Rollins.html) found in the CHILDES database. They are later tested and evaluated under a set of gold standard pairs.

To observe how the models perform when different types of stopword removal operations are applied, we have appended a static list of stopwords taken from NLTK, utilized a dynamic approach by taking word frequencies into account, and another set featuring a combination of those two. The evaluation results can be found [here](https://docs.google.com/spreadsheets/d/1i-z78_TrTZHpaurtagUqE_3dDRtiKL4LnZYdnfC5ras/edit?usp=sharing).
