import gzip
import pickle

with gzip.open("files/models/model.pkl.gz", "rb") as f:
    model = pickle.load(f)

print("Modelo cargado:", type(model))

if hasattr(model, "estimator"):
    print("Tipo de estimador dentro de GridSearchCV:", type(model.estimator))
    print("Pipeline Steps:", model.estimator.steps)