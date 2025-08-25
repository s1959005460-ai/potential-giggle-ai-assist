from core.registry import list_trainers, list_metrics
def test_registry_nonempty():
    # after importing defaults, registry should contain entries
    import trainer.register_defaults as r; import metrics.defaults as m
    assert 'gnn' in list_trainers()
    assert 'accuracy' in list_metrics()
