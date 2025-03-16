from matminer.featurizers.site import CrystalNNFingerprint
from matminer.featurizers.composition import ElementProperty

from .timeout import timeout

CompFP = ElementProperty.from_preset('magpie')

@timeout(5)
def timeout_featurize(structure, i):
    """Featureize a site in a structure."""
    return CrystalNNFingerprint.from_preset("ops").featurize(structure, i)