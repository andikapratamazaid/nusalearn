# Dataset sources

## iris.csv

The Iris flower dataset: 150 samples, 4 numeric features, 3 classes.

- Source: UCI Machine Learning Repository, Iris dataset
  (https://archive.ics.uci.edu/dataset/53/iris), file `iris.data`,
  downloaded directly from the UCI archive.
- Original reference: Fisher, R.A. (1936), "The use of multiple
  measurements in taxonomic problems", Annals of Eugenics 7(2).
- Data collected by Edgar Anderson (1935).
- The CSV in this repository adds a header row
  (sepal_length, sepal_width, petal_length, petal_width, target) to
  the original comma separated values. No values were changed.
- UCI distributes the dataset for research and education use; cite
  Fisher (1936) and the UCI repository when reusing it.

## wine.csv

The Wine recognition dataset: 178 samples, 13 numeric features,
3 classes (the class label is the first column, named `target` here).

- Source: UCI Machine Learning Repository, Wine dataset
  (https://archive.ics.uci.edu/dataset/109/wine), file `wine.data`,
  downloaded directly from the UCI archive.
- Original data: Forina, M. et al., Istituto di Analisi e Tecnologie
  Chimiche, University of Genova, Italy.
- The CSV adds a header row with the attribute names given in the
  UCI documentation (alcohol, malic_acid, ash, alcalinity_of_ash,
  magnesium, total_phenols, flavanoids, nonflavanoid_phenols,
  proanthocyanins, color_intensity, hue, od280_od315, proline).
  No values were changed.

## synthetic_regression.csv and synthetic_blobs.csv

These two files are SYNTHETIC. They are generated locally by
`make_synthetic.py` in this folder with a fixed random seed (42):

- synthetic_regression.csv: 300 samples, 3 features, target generated
  as a known linear combination plus Gaussian noise. Used by the
  regression tests and examples.
- synthetic_blobs.csv: 300 samples in 2 features drawn from three
  Gaussian blobs. Used by the clustering example and tests.

Because the generating process is known and fixed, results on these
files demonstrate correctness of the implementations only. They say
nothing about performance on real world data.
