
import numpy as np
from veroku.factors.gaussian import Gaussian
from veroku.factors.gaussian_mixture import GaussianMixture
from veroku.factors.gaussian_wishart import GaussianWishart
from veroku.factors.generlized_mixture import GeneralizedMixture


GAUS_CLASS_NAME = Gaussian.__class__.__name__
GAUS_MIX_CLASS_NAME = GaussianMixture.__class__.__name__
GUAS_WISH_CLASS_NAME = GaussianWishart.__class__.__name__
GEN_MIX_CLASS_NAME = GeneralizedMixture.__class__.__name__

def multiply(factor_a, factor_b):
    product = multiply_or_divide(factor_a, factor_b, operation_type="mulitply")
    return product
def multiply_or_divide(factor_a, factor_b, operation_type):
    assert operation_type == "multiply" or operation_type == "divide"

    do_multiplication = True
    if operation_type == "divide":
        do_multiplication = False

    if type(factor_a) == type(factor_b):
        return factor_a.absorb(factor_b)

    class_name_to_factor_dict = {
        factor_a.__class__.__name__: factor_a,
        factor_b.__class__.__name__: factor_b
    }
    factor_class_names = set(class_name_to_factor_dict.keys())
    if factor_class_names == {GAUS_CLASS_NAME, GAUS_MIX_CLASS_NAME}:
        gm = class_name_to_factor_dict[GAUS_MIX_CLASS_NAME]
        g = class_name_to_factor_dict[GAUS_CLASS_NAME]
        resulting_factors = []
        for component in gm:
            if do_multiplication:
                resulting_factor = g.absorb(component)
            else:
                resulting_factor = g.cancel(component)
            resulting_factors.append(resulting_factor)
        return GaussianMixture(resulting_factors)
    if GEN_MIX_CLASS_NAME in factor_class_names:
        other_factor_class_name = factor_class_names - {GEN_MIX_CLASS_NAME}
        gen_mix_factor = class_name_to_factor_dict[GEN_MIX_CLASS_NAME]
        other_factor = class_name_to_factor_dict[other_factor_class_name]
        resulting_factors = []
        for component in gen_mix_factor:
            product_factor = multiply(component, other_factor)
            resulting_factors.append(product_factor)
        return GeneralizedMixture(resulting_factors)






