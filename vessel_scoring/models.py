import vessel_scoring.legacy_heuristic_model
import vessel_scoring.random_forest_model
import vessel_scoring.logistic_model
import vessel_scoring.utils
import vessel_scoring.data
import vessel_scoring.add_measures
import vessel_scoring.colspec

import json
import os.path
import numpy
import importlib
import datetime

TRANSIT_WEIGHT = 10
all_data = ['kristina_trawl', 'kristina_longliner', 'kristina_ps'] + ['false_positives'] * TRANSIT_WEIGHT 
colspec={"windows": vessel_scoring.colspec.Colspec.windows}

untrained_models = {
    'Logistic':               {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6), 'data': all_data},

    'Logistic--Longliner':    {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6),
                               'data': ['kristina_longliner'] + ['false_positives'] * 10},
    'Logistic--Trawler':      {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6),
                               'data': ['kristina_trawl'] + ['false_positives'] * 10},

    # 'Logistic--Purse seine OLD': {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6),
    #                               'data': ['kristina_ps'] + ['false_positives'] * 10},

    'Logistic--Purse seine':  {
        'model': vessel_scoring.logistic_model.LogisticModel(
            colspec={
                "windows": vessel_scoring.colspec.Colspec.windows,
                "window_measures": vessel_scoring.colspec.Colspec.window_measures + ["measure_daylightavg"],
                "measures": vessel_scoring.colspec.Colspec.measures + ["measure_speed", "measure_daylight"]
                },
            order=3, cross=2),
        'data': ['kristina_ps'] + ['false_positives'] * 10},

    'Logistic opt MSE':       {'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=4, cross=3), 'data': all_data},
    'Random Forest':          {'model': vessel_scoring.random_forest_model.RandomForestModel(colspec=colspec), 'data': all_data},
    'Legacy':                 {'model': vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=3600), 'data': all_data},
    "Legacy (12 Hour)":       {'model': vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=43200), 'data': all_data},


    #### For notebooks/Compare models.ipynb ####
    'Logistic (SW)': {'active': False, 'compare_models': True, 'model': vessel_scoring.logistic_model.LogisticModel(colspec=dict(windows=[43200]), order=6), 'data': all_data},
    'Logistic (MW)': {'active': False, 'compare_models': True,
                      'model': vessel_scoring.logistic_model.LogisticModel(colspec=dict(windows=vessel_scoring.colspec.Colspec.windows), order=6),
                      'data': all_data},
    'Logistic (MW)': {'active': False, 'compare_models': True,
                      'model': vessel_scoring.logistic_model.LogisticModel(colspec=dict(windows=vessel_scoring.colspec.Colspec.windows), order=6),
                      'data': all_data},
    'Logistic (MW & daylight)': {'active': False, 'compare_models': True,
                                 'model': vessel_scoring.logistic_model.LogisticModel(colspec=dict(windows=vessel_scoring.colspec.Colspec.windows,
                                                                                                   measures=['measure_daylight']), order=6),
                                 'data': all_data},
    # Using `speed` here rather than `measure_speed` gives terrible results. Overflow? Just way too large?
    'Logistic (MW & daylight & speed)': {'active': False, 'compare_models': True,
                                         'model': vessel_scoring.logistic_model.LogisticModel(
                                             colspec=dict(windows=vessel_scoring.colspec.Colspec.windows,
                                                          measures=['measure_daylight', 'measure_speed']), order=6),
                                         'data': all_data},
    'Random Forest': {'active': False, 'compare_models': True,
                      'model': vessel_scoring.random_forest_model.RandomForestModel(colspec=dict(windows=[43200])),
                      'data': all_data},
    'Random Forest (MW)': {'active': False, 'compare_models': True,
                           'model': vessel_scoring.random_forest_model.RandomForestModel(colspec=dict(windows=vessel_scoring.colspec.Colspec.windows)),
                           'data': all_data},
    'Random Forest (MW & daylight)': {'active': False, 'compare_models': True,
                                      'model': vessel_scoring.random_forest_model.RandomForestModel(colspec=dict(windows=vessel_scoring.colspec.Colspec.windows,
                                                                                                                 measures=['measure_daylight'])),
                                      'data': all_data},
    'Random Forest (MW & daylight & speed)': {'active': False, 'compare_models': True, 'model': vessel_scoring.random_forest_model.RandomForestModel
                                              (colspec=dict(windows=vessel_scoring.colspec.Colspec.windows,
                                                            measures=['measure_daylight', 'speed'])),
                                              'data': all_data},


    #### Old models ####
    'Logistic (MW/cross3)': {'active': False, 'model': vessel_scoring.logistic_model.LogisticModel(colspec=colspec, order=6, cross=2), 'data': all_data},
    'Random Forest (MW)':   {'active': False,
                             'model': vessel_scoring.random_forest_model.RandomForestModel(colspec=dict(windows=vessel_scoring.colspec.Colspec.windows)),
                             'data': all_data},
    'Legacy (3 Hour)':      {'active': False, 'model': vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=10800), 'data': all_data},
    'Legacy (24 Hour)':     {'active': False, 'model': vessel_scoring.legacy_heuristic_model.LegacyHeuristicModel(window=86400),   'data': all_data},
}

def load_data(dir = None):
    if dir is None:
        dir = os.path.join(os.path.dirname(__file__), '..', 'datasets/data/labeled')

    datasets = {}
    for filename in os.listdir(dir):
        if filename.endswith('.measures.npz'):
            name = filename[:-len('.measures.npz')]
            datasets[name] = dict(zip(['all', 'train', 'cross', 'test'], vessel_scoring.data.load_dataset_by_vessel(os.path.join(dir, filename))))
    return datasets



def train_model_on_data(model, train_data):
    """train `model` with `train_data`

    example:

    model = train_model_on_data(LogisticModel(colsec=dict(window=3600)), train_data)


    """
    y_train = vessel_scoring.utils.is_fishy(train_data)
    model.fit(train_data, y_train)
    return model

def train_model(name, spec, dataset):
    print "Training %s..." % name
    training_data = ([dataset[name]['train']
                      for name in spec['data']]
                     + [dataset[name]['cross']
                        for name in spec['data']])
    training_data = vessel_scoring.utils.concatenate_different_recarrays(training_data)
    return train_model_on_data(spec['model'], training_data)

models_path = os.path.join(os.path.dirname(__file__), "models")

def train_models(models = None, train = None, save=True):
    if models is None:
        models = untrained_models
    if train is None:
        train = load_data()
    trained_models = {name: train_model(name, spec, train)
                      for (name, spec) in models.iteritems()
                      if spec.get("active", True)}
    if save:
        if not os.path.exists(models_path):
            os.mkdir(models_path)
        for (name, model) in trained_models.iteritems():
            res = model.dump_dict()
            if res is not None:
                with open(os.path.join(models_path, "%s.json" % name), "w") as f:
                    json.dump(res, f)
    return trained_models


def load_models():
    res = {}

    for filename in os.listdir(models_path):
        with open(os.path.join(models_path, filename)) as f:
            conf = json.load(f)
        name = filename.rsplit(".", 1)[0]

        modulename, clsname = conf['model'].rsplit('.', 1)
        modelcls = getattr(importlib.import_module(modulename), clsname)

        res[name] = modelcls(**conf['args'])
    return res
